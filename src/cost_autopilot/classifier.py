"""Phase 2.3 + 3.4: train, load and retrain the complexity classifier."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import settings
from .features import FEATURE_NAMES, feature_vector, heuristic_tier, split_instruction


def _hand_features(prompts):
    return np.array([feature_vector(p) for p in prompts])


def _instructions(prompts):
    return [split_instruction(p)[0] for p in prompts]


def build_pipeline(kind: str = "logreg") -> Pipeline:
    feats = FeatureUnion([
        ("hand", Pipeline([("f", FunctionTransformer(_hand_features)), ("s", StandardScaler())])),
        ("tfidf", Pipeline([("i", FunctionTransformer(_instructions)),
                            ("v", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True))])),
    ])
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("features", feats), ("clf", clf)])


def load_labeled(path: Path | None = None) -> tuple[list[str], list[int]]:
    X, y = [], []
    for line in Path(path or settings.LABELED_PROMPTS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            X.append(r["prompt"])
            y.append(int(r["tier"]))
    return X, y


def train(X: list[str] | None = None, y: list[int] | None = None, *, kind: str = "logreg",
          extra: list[tuple[str, int]] | None = None, out: Path | None = None,
          seed: int = 42) -> dict:
    """Train on labeled data (+ feedback examples), report held-out metrics,
    then refit on everything and save."""
    if X is None:
        X, y = load_labeled()
    X, y = list(X), list(y)
    extra = extra or []
    # feedback examples override the label of an identical prompt
    if extra:
        idx = {p: i for i, p in enumerate(X)}
        for p, t in extra:
            if p in idx:
                y[idx[p]] = t
            else:
                X.append(p)
                y.append(t)
                idx[p] = len(X) - 1

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    pipe = build_pipeline(kind).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    labels = [1, 2, 3]
    cm = confusion_matrix(yte, pred, labels=labels)
    # under-routing = predicted tier lower than true tier (the costly-quality mistake)
    under = int(sum(cm[i][j] for i in range(3) for j in range(3) if j < i))
    report = {
        "kind": kind,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train": len(Xtr), "n_test": len(Xte), "n_feedback": len(extra),
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "under_routing_rate": round(under / len(yte), 4),
        "confusion_matrix": {"labels": labels, "rows_true_cols_pred": cm.tolist()},
        "per_class": classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0),
        "features": FEATURE_NAMES,
    }
    cv = cross_val_score(build_pipeline(kind), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    report["cv5_accuracy_mean"] = round(float(cv.mean()), 4)
    report["cv5_accuracy_std"] = round(float(cv.std()), 4)
    final = build_pipeline(kind).fit(X, y)
    out = Path(out or settings.CLASSIFIER_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "report": report}, out)
    settings.CLASSIFIER_REPORT.parent.mkdir(parents=True, exist_ok=True)
    settings.CLASSIFIER_REPORT.write_text(json.dumps(report, indent=2))
    return report


@dataclass
class Prediction:
    tier: int
    confidence: float
    probabilities: dict[int, float]
    source: str  # "classifier" | "heuristic"


class ComplexityClassifier:
    """Loads the saved model and hot-reloads it when the file changes
    (so a retrain by the worker is picked up by the API without a restart)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.CLASSIFIER_PATH)
        self._mtime = None
        self._pipe = None

    def _maybe_reload(self):
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            self._pipe = None
            return
        if m != self._mtime:
            self._pipe = joblib.load(self.path)["pipeline"]
            self._mtime = m

    @property
    def ready(self) -> bool:
        self._maybe_reload()
        return self._pipe is not None

    def predict(self, prompt: str) -> Prediction:
        self._maybe_reload()
        if self._pipe is None:
            t = heuristic_tier(prompt)
            return Prediction(t, 0.5, {t: 0.5}, "heuristic")
        proba = self._pipe.predict_proba([prompt])[0]
        classes = [int(c) for c in self._pipe.classes_]
        probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        tier = max(probs, key=probs.get)
        return Prediction(tier, probs[tier], probs, "classifier")
