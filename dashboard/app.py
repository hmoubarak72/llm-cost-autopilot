"""Phase 4.2-4.3: cost and quality dashboard.  Run: streamlit run dashboard/app.py"""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cost_autopilot import settings
from cost_autopilot.analytics import daily, load
from cost_autopilot.db import DB
from cost_autopilot.registry import Registry

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
BLUE, GREY, ORANGE = "#2E6FD8", "#9AA5B1", "#E8833A"

req, esc, retrains = load()
st.title("LLM Cost Autopilot")
if settings.MOCK:
    st.warning("Simulated data (AUTOPILOT_MOCK=1): plumbing demo, not real quality or savings.")
if req.empty:
    st.info("No requests logged yet. Send traffic to POST /v1/completions or run `autopilot loadtest`.")
    st.stop()

freq = st.sidebar.radio("Group by", ["Day", "Week", "Hour"], index=0)
freq_code = {"Day": "D", "Week": "W", "Hour": "h"}[freq]
s = DB().stats()

# --- the money shot -------------------------------------------------------
st.markdown(
    f"<div style='font-size:64px;font-weight:700;color:{BLUE};line-height:1'>"
    f"{s['net_cost_reduction_pct']}%</div>"
    f"<div style='font-size:20px;margin-bottom:18px'>lower cost than sending everything to "
    f"<b>{Registry.load().baseline_name}</b>, "
    f"net of verification · <b>{s['quality_parity_pct']}%</b> quality parity on verified requests</div>",
    unsafe_allow_html=True)

c = st.columns(5)
c[0].metric("You saved", f"${s['net_saved_usd']:,.2f}")
c[1].metric("Total spend", f"${s['total_spend_usd']:,.2f}", help="routed calls + verification")
c[2].metric("Baseline would cost", f"${s['baseline_cost_usd']:,.2f}")
c[3].metric("Requests", f"{s['requests']:,}")
c[4].metric("Escalation rate", f"{s['escalation_rate_of_verified_pct'] or 0}%", help="of verified requests")

# --- cost over time -------------------------------------------------------
d = daily(req, freq_code)
left, right = st.columns([3, 2])
with left:
    st.subheader(f"Cost per {freq.lower()} vs all-baseline")
    fig = go.Figure()
    fig.add_bar(x=d.index, y=d["baseline_cost"], name="All on baseline", marker_color=GREY)
    fig.add_bar(x=d.index, y=d["total_cost"], name="Autopilot", marker_color=BLUE)
    fig.update_layout(barmode="group", height=340, margin=dict(t=10, b=10), yaxis_title="USD",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Routing distribution")
    dist = req["routed_model"].value_counts().reset_index()
    dist.columns = ["model", "requests"]
    fig = px.pie(dist, names="model", values="requests", hole=0.45,
                 color_discrete_sequence=[BLUE, ORANGE, "#3BA272", "#7A5AF8", GREY])
    fig.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Quality score distribution")
    v = req[req["verify_status"] == "done"]
    if v.empty:
        st.caption("No verified requests yet; is the worker running?")
    else:
        fig = px.histogram(v, x="quality_score", color="task_type", nbins=10, range_x=[0, 1],
                           labels={"quality_score": "score vs reference (0-1)"})
        fig.update_layout(height=320, margin=dict(t=10, b=10), bargap=0.05)
        st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Escalation rate over time")
    fig = px.line(d.reset_index(), x="time", y="escalation_rate", markers=True)
    fig.update_traces(line_color=ORANGE)
    fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_tickformat=".0%",
                      yaxis_range=[0, max(0.1, float(d["escalation_rate"].max()) * 1.3)])
    st.plotly_chart(fig, width="stretch")

st.subheader("Cost by model")
st.dataframe(req.groupby("routed_model").agg(requests=("id", "size"), cost=("cost", "sum"),
                                             avg_latency_ms=("latency_ms", "mean")).round(5),
             width="stretch")

if not esc.empty:
    st.subheader("Recent escalations")
    st.dataframe(esc.sort_values("ts", ascending=False)
                 [["time", "mode", "task_type", "original_model", "escalated_model", "quality_score",
                   "quality_gap", "cost_delta"]].head(50), width="stretch", hide_index=True)
if not retrains.empty:
    st.subheader("Classifier retrains (feedback loop)")
    st.dataframe(retrains[["time", "n_feedback", "accuracy", "cv_accuracy", "promoted"]], hide_index=True)

with st.expander("Audit trail (latest 200 requests)"):
    st.dataframe(req.sort_values("ts", ascending=False)
                 [["time", "prompt_hash", "task_type", "tier", "confidence", "routed_model", "cost",
                   "baseline_cost", "latency_ms", "verify_status", "quality_score", "escalated"]].head(200),
                 width="stretch", hide_index=True)
