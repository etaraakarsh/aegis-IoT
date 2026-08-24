#!/usr/bin/env python3
"""
07_analysis.py — tables and paired statistics.

Judge on the RELATIVE column, not the stars. With a large test partition,
differences far too small to matter operationally reach p<0.05.

    python scripts/07_analysis.py --results data/ton_iot/results/
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aegis.pipeline import paired_bootstrap

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--reference", default="exhaustive")
a = ap.parse_args()

raw = pd.read_csv(os.path.join(a.results, "main_results_raw.csv"))
num = raw.select_dtypes(include=[np.number]).columns
g = raw.groupby("condition")[num].mean()
sd = raw.groupby("condition")[num].std().fillna(0.0)

print("\n" + "-" * 96)
print("TABLE 1 — Main results (mean +/- sd over seeds)")
print("-" * 96)
print(f"{'condition':<16}{'macro-F1':<20}{'binary F1':<12}{'base':<9}"
      f"{'lift':<10}{'tools':<8}{'tokens':<9}{'cost':<8}")
print("-" * 96)
for c in g.index:
    print(f"{c:<16}{g.loc[c,'macro_f1']:.4f} +/- {sd.loc[c,'macro_f1']:.4f}   "
          f"{g.loc[c,'binary_f1']:<12.4f}{g.loc[c,'binary_majority_baseline']:<9.4f}"
          f"{g.loc[c,'binary_lift']:<+10.4f}{g.loc[c,'mean_n_tools']:<8.1f}"
          f"{g.loc[c,'mean_tokens']:<9.0f}{g.loc[c,'mean_total_cost']:<8.2f}")

P = json.load(open(os.path.join(a.results, "predictions.json")))
truth, preds = P["truth"], P["predictions"]
ref = a.reference if a.reference in preds else list(preds)[0]

print("\n" + "-" * 96)
print(f"TABLE 2 — Paired macro-F1 difference vs '{ref}'")
print("-" * 96)
print("  n is large, so tiny gaps reach p<0.05. Judge on the relative column.")
print(f"\n{'condition':<16}{'diff':<11}{'95% CI':<24}{'p':<10}{'rel%':<9}verdict")
print("-" * 96)
out = []
for c, yp in preds.items():
    if c == ref:
        continue
    r = paired_bootstrap(truth, yp, preds[ref], n=1000)
    base = float(g.loc[ref, "macro_f1"]) or 1.0
    rel = 100.0 * r["diff"] / base
    v = ("practically equivalent" if abs(rel) < 1 else
         "moderate" if abs(rel) < 20 else "LARGE")
    if r["p"] >= 0.05:
        v += " (n.s.)"
    print(f"{c:<16}{r['diff']:<+11.4f}"
          f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]     "
          f"{r['p']:<10.4f}{rel:<+9.1f}{v}")
    out.append({"condition": c, **r, "relative_pct": rel})
pd.DataFrame(out).to_csv(os.path.join(a.results, "paired_tests.csv"), index=False)

pc = pd.read_csv(os.path.join(a.results, "per_class_f1.csv"), index_col=0)
print("\n" + "-" * 96)
print("TABLE 3 — Per-class F1")
print("-" * 96)
print(pc.round(4).to_string())

# rare-class sensitivity
if "constrained" in pc.columns and "exhaustive" in pc.columns:
    import os as _os
    dd = _os.path.dirname(a.results.rstrip("/"))
    tr = _os.path.join(dd, "train.csv")
    if _os.path.exists(tr):
        freq = pd.read_csv(tr, usecols=["true_type"]).true_type.value_counts()
        print("\n" + "-" * 96)
        print("TABLE 3b — Where capacity reduction bites (rare classes first)")
        print("-" * 96)
        print(f"{'class':<16}{'train n':>10}{'exhaustive':>13}"
              f"{'constrained':>13}{'abs loss':>11}{'rel loss %':>12}")
        print("-" * 96)
        rows = []
        for c in pc.index:
            if c not in freq:
                continue
            e, k = float(pc.loc[c, "exhaustive"]), float(pc.loc[c, "constrained"])
            rows.append((c, int(freq[c]), e, k, e - k,
                         100.0 * (e - k) / max(e, 1e-9)))
        for c, n, e, k, dl, rel in sorted(rows, key=lambda r: r[1]):
            print(f"{c:<16}{n:>10,}{e:>13.4f}{k:>13.4f}{dl:>+11.4f}{rel:>+12.1f}")
        if rows:
            rare = min(rows, key=lambda r: r[1])
            common = [r for r in rows if r[1] > 2 * rare[1]]
            if common:
                avg = sum(r[5] for r in common) / len(common)
                print(f"\n  rarest class '{rare[0]}' (n={rare[1]:,}) loses "
                      f"{rare[5]:+.1f}% relative;")
                print(f"  classes with >2x its frequency lose {avg:+.1f}% on average.")
                if rare[5] > 2 * max(abs(avg), 0.5):
                    print("  -> Capacity reduction costs the rarest class several times")
                    print("     what it costs common ones. Report this: it is the")
                    print("     operational limit on how far the floor can be relaxed.")

print("\n" + "-" * 96)
print("TABLE 4 — Cost decomposition (proves no cost-shifting)")
print("-" * 96)
print(f"{'condition':<16}{'tool':<10}{'token':<10}{'total':<10}"
      f"{'token share':<14}{'tokens':<10}{'latency s':<10}")
print("-" * 96)
for c in g.index:
    t, k, tot = (g.loc[c, 'mean_tool_cost'], g.loc[c, 'mean_token_cost'],
                 g.loc[c, 'mean_total_cost'])
    print(f"{c:<16}{t:<10.3f}{k:<10.3f}{tot:<10.3f}{(k/tot if tot else 0):<14.3f}"
          f"{g.loc[c,'mean_tokens']:<10.0f}{g.loc[c,'mean_latency_s']:<10.3f}")
print("\nA low-tool-cost condition with a high token share is SHIFTING cost,")
print("not saving it. Any efficiency claim must cite the TOTAL column.")

# injection: say plainly when it is a null result
inj = {}
for sub in ("in_dist", "held_out"):
    p = os.path.join(a.results, sub, "injection_summary.json")
    if os.path.exists(p):
        inj[sub] = json.load(open(p))
if inj:
    print("\n" + "-" * 96)
    print("TABLE 5 — Indirect prompt injection")
    print("-" * 96)
    print(f"{'payload set':<20}{'arm':<12}{'verdicts changed':>19}"
          f"{'d FN':>10}{'d FP':>10}")
    print("-" * 96)
    allzero = True
    for sub, d in inj.items():
        for arm in ("injected_vs_control", "defended_vs_control"):
            v = d[arm]
            if abs(v["verdict_type_changed_frac"]) > 1e-9:
                allzero = False
            print(f"{d['payload_set']:<20}{arm.split('_')[0]:<12}"
                  f"{100*v['verdict_type_changed_frac']:>18.2f}%"
                  f"{v['delta_fn_rate']:>+10.4f}{v['delta_fp_rate']:>+10.4f}")
    if allzero:
        print("\n  NULL RESULT. No verdict changed in any arm, either payload set.")
        print("  Report it as such: the attack did not move verdicts here and the")
        print("  defense had nothing to prevent. You may NOT claim the defense")
        print("  works, and you may NOT claim evidence strength is the reason")
        print("  without a matching WEAK-EVIDENCE run to compare against.")
        print("  Run the telemetry corpus and pass --compare-dir to 08_figures.py.")

pol = json.load(open(os.path.join(a.results, "policy.json")))
print(f"\ncost provenance: {pol.get('cost_provenance')}   fusion: {pol.get('fusion')}")
if pol.get("cost_provenance") != "measured":
    print("  STOP: costs are not measured. Do not report cost results.")
if pol.get("fusion") == "deterministic":
    print("  NOTE: deterministic fusion -> token cost is 0 by construction.")
    print("  Say so, or re-run with --fusion llm before any agentic claim.")
print("\nnext:  python scripts/08_figures.py --results", a.results)
