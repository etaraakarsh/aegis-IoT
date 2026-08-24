#!/usr/bin/env python3
"""
00_make_synthetic.py — a fake corpus with TON_IoT's shape, for smoke-testing
the pipeline before you download 16 GB. Results from it are meaningless.
"""
import argparse, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=12000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
classes = ["normal","backdoor","ddos","injection","password","ransomware","scanning","xss"]
groups = ["net_other","net_dns","net_http"]
rows = []
for i in range(a.n):
    c = rng.choice(classes, p=[.30]+[.10]*7)
    g = rng.choice(groups, p=[.6,.2,.2])
    sig = classes.index(c)
    r = {"true_type": c, "device": g, "domain": g}
    for j in range(24):
        base = rng.normal(0, 1)
        if j < 8:
            base += sig * 0.85 * rng.normal(1, .25)   # informative
        elif j < 14:
            base += (sig % 3) * 0.4                    # weakly informative
        r[f"f{j}"] = base
    rows.append(r)
df = pd.DataFrame(rows)
os.makedirs(a.out, exist_ok=True)
df.to_csv(os.path.join(a.out, "prepared.csv"), index=False)
print(f"wrote {a.out}/prepared.csv  ({len(df):,} synthetic rows) — SMOKE TEST ONLY")
