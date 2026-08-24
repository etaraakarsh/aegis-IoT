#!/usr/bin/env python3
"""
02_build_eval_set.py — rebalance and split.

Attack families are DOWNSAMPLED to reach the benign target; benign flows are
never synthesised. Oversampling would place fabricated rows into a partition
meant to measure behaviour on real traffic.

The majority-class binary F1 is printed for every split. Quote the TEST figure
beside every binary number you report -- it is the bar.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

a = argparse.ArgumentParser()
a.add_argument("--data-dir", required=True)
a.add_argument("--normal-frac", type=float, default=0.35)
a.add_argument("--min-per-class", type=int, default=30)
a.add_argument("--allow-unequal", action="store_true",
               help="let each family take min(cap, its size) instead of forcing "
                    "every family down to the smallest. Keeps genuinely rare "
                    "classes (TON_IoT mitm, Bot-IoT theft) without collapsing "
                    "the pool. RECOMMENDED.")
a.add_argument("--drop-below", type=int, default=0,
               help="exclude attack families with fewer than N records")
a.add_argument("--min-group-rows", type=int, default=400,
               help="minimum rows a group needs in the REBALANCED pool")
a.add_argument("--seed", type=int, default=0)
a = a.parse_args()

df = pd.read_csv(os.path.join(a.data_dir, "prepared.csv"), low_memory=False)
rng = np.random.default_rng(a.seed)
print("\nsource distribution:")
print(df.true_type.value_counts().to_string())

n_norm = int((df.true_type == "normal").sum())
fams = [c for c in df.true_type.unique() if c != "normal"]
if not fams:
    sys.exit("no attack classes found")

if a.drop_below:
    small = [f for f in fams if int((df.true_type == f).sum()) < a.drop_below]
    if small:
        print(f"\ndropping families under {a.drop_below}: {small}")
        df = df[~df.true_type.isin(small)]
        fams = [f for f in fams if f not in small]

# benign is almost always the limiting resource; cap attacks to hit the target
cap = int(round(n_norm * (1 - a.normal_frac) / (a.normal_frac * len(fams))))
smallest = min(int((df.true_type == f).sum()) for f in fams)
if a.allow_unequal:
    # Each family takes what it has, up to the cap. A 1,043-record class stays
    # rare rather than forcing all nine families down to 1,043, which would
    # discard ~90% of the usable data to accommodate one class.
    sizes = {f: min(cap, int((df.true_type == f).sum())) for f in fams}
    rare = [f for f, n in sizes.items() if n < cap]
    if rare:
        print(f"\nunequal caps: {rare} stay at natural size; others capped at {cap:,}")
        print("  Rare classes are exactly what a cost-driven policy abandons first,")
        print("  so keeping them makes the ablation more informative, not less.")
else:
    cap = min(cap, smallest)
    sizes = {f: cap for f in fams}
if min(sizes.values()) < a.min_per_class:
    sys.exit(f"\nSTOP: smallest family would be {min(sizes.values())} "
             f"(< {a.min_per_class}).\n"
             f"Only {n_norm} benign flows are available. Use a larger release\n"
             f"of this corpus, drop the smallest attack family, or lower\n"
             f"--normal-frac (and report the change).\n")

parts = [df[df.true_type == f].sample(sizes[f], random_state=a.seed) for f in fams]
n_attack = sum(sizes.values())
n_need = int(round(n_attack * a.normal_frac / (1 - a.normal_frac)))
parts.append(df[df.true_type == "normal"].sample(min(n_need, n_norm),
                                                 random_state=a.seed))
pool = pd.concat(parts).sample(frac=1.0, random_state=a.seed).reset_index(drop=True)

# Re-check group viability on the REBALANCED pool, not the source corpus.
if "device" in pool.columns:
    need = max(a.min_group_rows, 120)
    counts = pool.device.value_counts()
    small = [g for g, n in counts.items() if n < need]
    if small and len(counts) > len(small):
        biggest = counts.drop(labels=small).idxmax()
        print(f"\nafter rebalancing, these groups are too small for their own "
              f"detectors: {[f'{g}(n={counts[g]})' for g in small]}")
        print(f"  folding into {biggest}. A group with no fitted detectors "
              f"produces no evidence,")
        print( "  so every one of its incidents would default to benign.")
        pool.loc[pool.device.isin(small), ["device", "domain"]] = biggest
        print(f"  groups now: {dict(pool.device.value_counts())}")
frac = float((pool.true_type == "normal").mean())
print(f"\nbalanced pool: n={len(pool):,}  benign={frac:.3f}")
print("  per family: " + ", ".join(f"{f}={sizes[f]:,}" for f in sorted(sizes)))

n = len(pool); i1, i2 = int(0.5 * n), int(0.7 * n)
splits = {"train": pool.iloc[:i1], "val": pool.iloc[i1:i2], "test": pool.iloc[i2:]}
meta = {"benign_fraction": frac, "per_family_sizes": sizes,
        "allow_unequal": bool(a.allow_unequal), "seed": a.seed}
for k, s in splits.items():
    yt = (s.true_type != "normal").astype(int)
    maj = 1 if yt.sum() >= len(yt) - yt.sum() else 0
    from sklearn.metrics import f1_score
    b = f1_score(yt, [maj] * len(yt), zero_division=0)
    s.to_csv(os.path.join(a.data_dir, f"{k}.csv"), index=False)
    meta[f"{k}_n"] = len(s); meta[f"{k}_majority_binary_f1"] = round(float(b), 4)
    print(f"  {k:<6} n={len(s):<7,} benign={float((s.true_type=='normal').mean()):.3f}"
          f"  majority-baseline binF1={b:.4f}")

json.dump(meta, open(os.path.join(a.data_dir, "split_meta.json"), "w"), indent=2)
print(f"\nQuote {meta['test_majority_binary_f1']:.4f} beside every binary number.")
print("next:  python scripts/03_measure_costs.py --data-dir", a.data_dir)
