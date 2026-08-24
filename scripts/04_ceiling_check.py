#!/usr/bin/env python3
"""
04_ceiling_check.py — what is achievable before any agent is involved.

Read every agent result as a fraction of this number, not of 1.0.

IMPORTANT, and the reason this script is separate. There are two different
statistics people call "the ceiling":

  POOLED   fit per-group models, concatenate all predictions, compute ONE
           macro-F1 over the pooled result.
  AVERAGED fit per-group models, compute macro-F1 within each group, then take
           a weighted mean.

They are not the same and can differ by eight points, because groups contain
different class subsets and averaging silently reweights the classes. The agent
pipeline reports POOLED. Comparing an agent's pooled score against an averaged
ceiling makes the agent appear to exceed what is achievable, which is
impossible and should be read as a bug rather than a result. This script prints
both and labels which one is comparable.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aegis.data import live_features


def fit_predict(tr, te, cols, seed=0, binary=False):
    if len(tr) < 30 or len(te) < 10 or len(cols) < 2:
        return None, None
    sc = StandardScaler().fit(tr[cols].to_numpy(float))
    ytr = (tr.true_type != "normal").astype(int) if binary else tr.true_type
    yte = (te.true_type != "normal").astype(int) if binary else te.true_type
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                               class_weight="balanced_subsample",
                               n_jobs=-1, random_state=seed
                               ).fit(sc.transform(tr[cols].to_numpy(float)), ytr)
    return list(yte), list(m.predict(sc.transform(te[cols].to_numpy(float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr = pd.read_csv(os.path.join(a.data_dir, "train.csv"), low_memory=False)
    te = pd.read_csv(os.path.join(a.data_dir, "test.csv"), low_memory=False)
    meta = ["true_type", "device", "domain"]
    feats = [c for c in tr.columns if c not in meta]

    print("\n" + "-" * 74)
    print("CEILING CHECK — best achievable macro-F1, no agents involved")
    print("-" * 74)
    print(f"train n={len(tr):,}  test n={len(te):,}  features={len(feats)}")

    yt, yp = fit_predict(tr, te, live_features(tr, feats), a.seed)
    glob = f1_score(yt, yp, average="macro", zero_division=0)
    print(f"\n[A] GLOBAL, all features pooled           macro-F1 = {glob:.4f}")

    print("\n[B] PER-GROUP")
    pool_t, pool_p, scores, weights = [], [], [], []
    for g in sorted(tr.device.unique()):
        t, e = tr[tr.device == g], te[te.device == g]
        cols = live_features(t, feats)
        yt, yp = fit_predict(t, e, cols, a.seed)
        if not yt:
            continue
        s = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"      {g:<18} feats={len(cols):<3} n_test={len(e):<6} "
              f"classes={e.true_type.nunique()}  macro-F1 = {s:.4f}")
        pool_t += yt; pool_p += yp
        scores.append(s); weights.append(len(e))

    pooled = f1_score(pool_t, pool_p, average="macro", zero_division=0)
    averaged = float(np.average(scores, weights=weights)) if scores else 0.0

    yt, yp = fit_predict(tr, te, live_features(tr, feats), a.seed, binary=True)
    binf1 = f1_score(yt, yp, zero_division=0)

    print("\n" + "-" * 74)
    print("VERDICT")
    print("-" * 74)
    print(f"  POOLED per-group   = {pooled:.4f}   <-- COMPARE AGENTS TO THIS")
    print(f"  AVERAGED per-group = {averaged:.4f}   (different statistic, NOT")
    print( "                                       comparable to agent scores)")
    print(f"  GLOBAL single model= {glob:.4f}")
    print(f"  BINARY task        = {binf1:.4f}")
    ceiling = max(pooled, glob)
    print(f"\n  -> Report {ceiling:.4f} as the achievable macro-F1 ceiling.")
    print(f"  -> Read every agent result as a fraction of {ceiling:.4f}, not 1.0.")
    if averaged < pooled - 0.02:
        print(f"\n  NOTE: averaged is {pooled - averaged:.4f} below pooled. Quoting the")
        print( "  averaged figure would make your agent look better than possible.")

    import json
    json.dump({"pooled": float(pooled), "averaged": float(averaged),
               "global": float(glob), "binary": float(binf1),
               "ceiling_to_report": float(ceiling)},
              open(os.path.join(a.data_dir, "ceiling.json"), "w"), indent=2)
    print("\nnext:  python scripts/05_run_conditions.py --data-dir", a.data_dir)


if __name__ == "__main__":
    main()
