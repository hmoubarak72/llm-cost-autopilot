from fastapi.testclient import TestClient

from cost_autopilot.api import app


def test_completion_endpoint(trained):
    c = TestClient(app)
    r = c.post("/v1/completions", json={"messages": [{"role": "user", "content": "Uppercase: hello world"}],
                                        "model": "ignored"})
    assert r.status_code == 200
    j = r.json()
    assert j["choices"][0]["message"]["content"]
    rt = j["routing"]
    assert rt["selected_model"] and rt["reason"] and rt["tier"] in (1, 2, 3)
    assert rt["baseline_model"] == "gpt-4o"
    assert rt["saved_usd"] == round(rt["baseline_cost_usd"] - rt["cost_usd"], 8)
    row = c.get(f"/v1/requests/{rt['request_id']}").json()
    assert row["routed_model"] == rt["served_by"]


def test_validation_errors(trained):
    c = TestClient(app)
    assert c.post("/v1/completions", json={}).status_code == 422
    assert c.put("/v1/routing-config", json={"tiers": {"1": "nope"}}).status_code == 422


def test_config_endpoints(trained):
    c = TestClient(app)
    models = c.get("/v1/models").json()
    assert any(m["name"] == "gpt-4o-mini" for m in models["models"])
    r = c.put("/v1/routing-config", json={"tiers": {"1": "claude-haiku"}})
    assert r.status_code == 200 and r.json()["tiers"]["1"] == "claude-haiku"
    j = c.post("/v1/completions", json={"prompt": "Uppercase: hello world"}).json()
    if j["routing"]["tier"] == 1:
        assert j["routing"]["served_by"] == "claude-haiku"
    s = c.get("/v1/stats").json()
    assert s["requests"] >= 1
    assert c.get("/health").json()["ok"]
