from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

TIERS = {"lite": 5, "mid": 12, "full": None}      # None == all live features


@dataclass
class Tool:
    name: str
    group: str
    kind: str                  # classifier | isoforest | rules | drift
    tier: str = ""
    n_features: int = 0
    model: object = None
    scaler: object = None
    features: List[str] = field(default_factory=list)
    classes_: np.ndarray = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Unnormalised outlier score for unsupervised tools."""
        return -self.model.score_samples(self.scaler.transform(X))


class Registry:
    def __init__(self, seed: int = 0):
        self.tools: Dict[str, Tool] = {}
        self.seed = seed

    # fit
    def fit(self, train, feature_cols, verbose=True):
        from .data import live_features
        t0 = time.time()
        for grp in sorted(train.device.unique()):
            g = train[train.device == grp]
            live = live_features(g, feature_cols)
            if len(g) < 60 or len(live) < 3:
                continue
            X, y = g[live].to_numpy(float), g.true_type.to_numpy()

            # rank features once per group, reuse for every tier
            try:
                mi = mutual_info_classif(X, y, random_state=self.seed)
            except Exception:
                mi = np.var(X, axis=0)
            order = [live[i] for i in np.argsort(-mi)]

            for tier, k in TIERS.items():
                cols = order if k is None else order[:min(k, len(order))]
                if len(cols) < 2:
                    continue
                self._fit_clf(grp, tier, g, cols)

            self._fit_iso(grp, g, order[:min(12, len(order))])
            self._fit_rules(grp, g, order[:min(6, len(order))])
            if grp.endswith("other"):
                self._fit_drift(grp, g, order[:min(6, len(order))])

        if verbose:
            n_clf = sum(1 for t in self.tools.values() if t.kind == "classifier")
            print(f"  fitted {len(self.tools)} tools "
                  f"({n_clf} classifiers across {len(TIERS)} tiers) "
                  f"in {time.time() - t0:.1f}s")
        return self

    def _fit_clf(self, grp, tier, g, cols):
        # Depth and ensemble size scale with tier so that cost differs for a
        # real reason rather than by fiat.
        n_est = {"lite": 60, "mid": 150, "full": 300}[tier]
        depth = {"lite": 8, "mid": 16, "full": None}[tier]
        sc = StandardScaler().fit(g[cols].to_numpy(float))
        m = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=self.seed,
        ).fit(sc.transform(g[cols].to_numpy(float)), g.true_type.to_numpy())
        # Fit in parallel, PREDICT single-threaded. joblib's thread dispatch
        # costs more than the tree traversal it parallelises at inference
        # scale, and it flattens the tier costs the whole design depends on:
        # on Apple Silicon the 60-tree lite model measured MORE expensive than
        # the 150-tree mid model. Single-threaded prediction restores the
        # 1 : 2.8 : 5.4 ratio the tiers are supposed to have.
        m.n_jobs = 1
        name = f"{grp}_clf_{tier}"
        self.tools[name] = Tool(name, grp, "classifier", tier, len(cols),
                                m, sc, cols, m.classes_)

    def _fit_iso(self, grp, g, cols):
        benign = g[g.true_type == "normal"]
        if len(benign) < 40:
            benign = g
        sc = StandardScaler().fit(benign[cols].to_numpy(float))
        m = IsolationForest(n_estimators=120, contamination=0.1,
                            random_state=self.seed, n_jobs=-1
                            ).fit(sc.transform(benign[cols].to_numpy(float)))
        m.n_jobs = 1
        name = f"{grp}_isoforest"
        self.tools[name] = Tool(name, grp, "isoforest", "", len(cols),
                                m, sc, cols)

    def _fit_rules(self, grp, g, cols):
        benign = g[g.true_type == "normal"]
        if len(benign) < 20:
            benign = g
        lo = benign[cols].quantile(0.01).to_numpy(float)
        hi = benign[cols].quantile(0.99).to_numpy(float)
        name = f"{grp}_rules"
        self.tools[name] = Tool(name, grp, "rules", "", len(cols),
                                (lo, hi), None, cols)

    def _fit_drift(self, grp, g, cols):
        benign = g[g.true_type == "normal"]
        if len(benign) < 20:
            benign = g
        mu = benign[cols].mean().to_numpy(float)
        sd = benign[cols].std().replace(0, 1).to_numpy(float)
        name = f"{grp}_drift"
        self.tools[name] = Tool(name, grp, "drift", "", len(cols),
                                (mu, sd), None, cols)

    # run
    def invoke(self, name: str, row: dict):
        """Return an evidence dict, or None when the tool does not apply."""
        t = self.tools.get(name)
        if t is None:
            return None
        try:
            x = np.array([[float(row.get(c, 0.0)) for c in t.features]])
        except Exception:
            return None

        if t.kind == "classifier":
            p = t.predict_proba(x)[0]
            i = int(np.argmax(p))
            return {"tool": name, "kind": "classifier", "tier": t.tier,
                    "label": str(t.classes_[i]), "confidence": float(p[i]),
                    "proba": {str(c): float(v) for c, v in zip(t.classes_, p)}}
        if t.kind == "isoforest":
            s = float(t.score(x)[0])
            return {"tool": name, "kind": "isoforest",
                    "label": "attack" if s > 0.55 else "normal",
                    "confidence": float(min(1.0, abs(s - 0.5) * 2))}
        if t.kind == "rules":
            lo, hi = t.model
            v = x[0]
            frac = float(np.mean((v < lo) | (v > hi)))
            return {"tool": name, "kind": "rules",
                    "label": "attack" if frac > 0.25 else "normal",
                    "confidence": float(min(1.0, frac * 2))}
        if t.kind == "drift":
            mu, sd = t.model
            z = float(np.mean(np.abs((x[0] - mu) / sd)))
            return {"tool": name, "kind": "drift",
                    "label": "attack" if z > 2.5 else "normal",
                    "confidence": float(min(1.0, z / 5.0))}
        return None

    def for_group(self, grp: str) -> List[str]:
        return [n for n, t in self.tools.items() if t.group == grp]

    def names(self) -> List[str]:
        return sorted(self.tools)
