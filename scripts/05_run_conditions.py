#!/usr/bin/env python3
"""
05_run_conditions.py — the main experiment.

    # free, no API calls, tool-compute costs only
    python scripts/05_run_conditions.py --data-dir data/ton_iot/ --seeds 5

    # the version to report: real LLM fusion, real token counts
    python scripts/05_run_conditions.py --data-dir data/ton_iot/ \
        --fusion llm --seeds 3 --limit 1500

    # re-run analysis on cached LLM responses, free
    python scripts/05_run_conditions.py --data-dir data/ton_iot/ --fusion replay

Conditions:
    exhaustive       every tool in the group
    domain_all       same (retained for continuity with v5 naming)
    fixed            one full-tier classifier per group, unmeasured
    random           uniform random subset -- sanity check
    core_only        the measured mandatory core alone
    constrained      core + budgeted extras by value/cost   <- the method
    no_core          cost-only minimisation                 <- the failure
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aegis.costs import load_costs
from aegis.llm import Fuser
from aegis.pipeline import (Pipeline, binary_scores, bootstrap_ci,
                            degeneracy_report, macro_f1, per_class_f1)
from aegis.policy import (ConstrainedPolicy, UnconstrainedPolicy, fixed_subset,
                          measure_marginal_values, prune_harmful_core,
                          select_core_by_tier)
from aegis.tools import Registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fusion", default="deterministic",
                    choices=["deterministic", "llm", "replay"])
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap test incidents (use with --fusion llm)")
    ap.add_argument("--val-subsample", type=int, default=1200,
                    help="incidents used for marginal-value estimation")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="NCU budget; 0 = auto (core cost x 1.5)")
    ap.add_argument("--epsilon", type=float, default=0.02,
                    help="accuracy floor tolerance. 0.02 is the recommended "
                         "operating point: on TON_IoT it costs 0.07%% relative "
                         "macro-F1 for a 3.3x cost reduction, whereas 0.05 "
                         "costs 1.5%% for 4.7x. Justify your choice with "
                         "--epsilon-sweep rather than accepting the default.")
    ap.add_argument("--epsilon-sweep", action="store_true",
                    help="sweep the accuracy floor and trace the cost frontier")
    a = ap.parse_args()
    out = a.out or os.path.join(a.data_dir, "results")
    os.makedirs(out, exist_ok=True)

    costs = load_costs(a.data_dir)      # raises if not measured
    print(f"\ncost model: {costs.provenance}, {len(costs.tool_cost)} tools, "
          f"token={costs.token_cost_per_1k:.4f} NCU/1k")
    if costs.token_cost_per_1k == 0 and a.fusion == "llm":
        print("  WARNING: token cost is 0 but fusion is llm. Re-run")
        print("  03_measure_costs.py --time-llm or token spend stays invisible.")

    tr = pd.read_csv(os.path.join(a.data_dir, "train.csv"), low_memory=False)
    va = pd.read_csv(os.path.join(a.data_dir, "val.csv"), low_memory=False)
    te = pd.read_csv(os.path.join(a.data_dir, "test.csv"), low_memory=False)
    meta = ["true_type", "device", "domain"]
    feats = [c for c in tr.columns if c not in meta]
    classes = sorted(tr.true_type.unique())

    if a.limit and a.limit < len(te):
        te = te.sample(a.limit, random_state=0).reset_index(drop=True)
        print(f"  test capped at {len(te):,} incidents (--limit)")

    if a.fusion == "llm":
        n_cond = 7
        calls = len(te) * n_cond * a.seeds
        in_tok, out_tok = 550, 45           # measured from the prompt template
        usd = (calls * in_tok / 1e6) * 3.0 + (calls * out_tok / 1e6) * 15.0
        print("\n" + "-" * 62)
        print("LLM RUN COST ESTIMATE (before any call is made)")
        print("-" * 62)
        print(f"  {len(te):,} incidents x {n_cond} conditions x {a.seeds} seeds"
              f" = {calls:,} calls")
        print(f"  ~{calls * (in_tok + out_tok) / 1e6:.1f}M tokens")
        print(f"  ~${usd:,.0f} at Sonnet list price, less any cache hits")
        print("  Ctrl-C now if that is more than you intended to spend.")
        print("-" * 62)
        time.sleep(6)

    reg = Registry(seed=0).fit(tr, feats)
    cache = os.path.join(out, "llm_cache.json")
    fuser = Fuser(mode=a.fusion, model=a.model, cache_path=cache)
    pipe = Pipeline(reg, fuser, costs, classes)

    # marginal values on validation
    vs = va.sample(min(a.val_subsample, len(va)), random_state=0).to_dict("records")
    det = Fuser(mode="deterministic")           # value estimation stays free
    est = Pipeline(reg, det, costs, classes)

    def evaluate_subset(tools, group=None):
        """Macro-F1 of a tool subset on validation. `group` restricts scoring
        to incidents from that group, which is required when comparing
        classifier tiers: a tier only fires on its own group's traffic."""
        rows = vs if group is None else [r for r in vs if r.get("device") == group]
        if not rows:
            return 0.0
        yt, yp = [], []
        for r in rows:
            avail = [t for t in tools if reg.tools.get(t)
                     and reg.tools[t].group == r.get("device", "")]
            res = est.run_one(r, avail)
            yt.append(r["true_type"]); yp.append(res["label"])
        return macro_f1(yt, yp)

    all_tools = reg.names()
    print(f"\nestimating marginal values on {len(vs)} validation incidents")
    core = select_core_by_tier(evaluate_subset, all_tools, costs, a.epsilon)
    core = prune_harmful_core(core, evaluate_subset)
    values = measure_marginal_values(evaluate_subset, all_tools, core)
    core_cost = costs.tools(core)
    budget = a.budget or core_cost * 1.5
    print(f"\n  mandatory core: {core}")
    print(f"  core cost = {core_cost:.2f} NCU   budget = {budget:.2f} NCU")

    pol = ConstrainedPolicy(core, values, costs, budget, a.epsilon)
    # The unconstrained policy must be given a budget it cannot satisfy with a
    # classifier, otherwise it simply buys everything and no failure occurs.
    # This is the whole point of the ablation: under real cost pressure a
    # cost-only objective abandons the expensive informative tools.
    cheapest_clf = min((c for t, c in costs.tool_cost.items()
                        if "_clf_" in t), default=budget)
    unc = UnconstrainedPolicy(costs, cheapest_clf * 0.9)
    print(f"  no-core ablation budget = {cheapest_clf * 0.9:.2f} NCU "
          f"(below cheapest classifier at {cheapest_clf:.2f})")

    def sel_core(avail):
        return [t for t in core if t in avail]

    def sel_exhaustive(av):
        """Every auxiliary tool plus the widest classifier tier. Invoking all
        three tiers is not 'more thorough', it is redundant."""
        return [t for t in av if "_clf_" not in t or t.endswith("_clf_full")]

    CONDITIONS = {
        "exhaustive": sel_exhaustive,
        "domain_all": sel_exhaustive,
        "fixed": fixed_subset,
        "random": None,                        # seeded per run below
        "core_only": sel_core,
        "constrained": pol.select,
        "no_core": unc.select,
    }

    rows, preds, pcs = [], {}, {}
    truth = list(te.true_type)
    recs = te.to_dict("records")

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        for name, fn in CONDITIONS.items():
            if name == "random":
                def fn(av, _r=rng):
                    if not av:            # group has no fitted tools
                        return []
                    k = int(_r.integers(1, len(av) + 1))
                    return list(_r.choice(av, size=min(k, len(av)), replace=False))
            t0 = time.time()
            res = []
            step = max(1, len(recs) // 10)
            for i, r in enumerate(recs):
                res.append(pipe.run_one(r, fn(reg.for_group(r.get("device", "")))))
                if i and i % step == 0:
                    el = time.time() - t0
                    print(f"      {name} seed={seed}  {100*i//len(recs):>3}%  "
                          f"{el:.0f}s elapsed, ~{el*(len(recs)-i)/i:.0f}s left",
                          flush=True)
            yp = [x["label"] for x in res]
            lo, hi = bootstrap_ci(truth, yp, n=500, seed=seed)
            b = binary_scores(truth, yp)
            deg = degeneracy_report(truth, yp, classes)
            row = {"condition": name, "seed": seed, "n": len(truth),
                   "macro_f1": macro_f1(truth, yp),
                   "macro_f1_ci_low": lo, "macro_f1_ci_high": hi, **b,
                   "mean_tool_cost": float(np.mean([x["tool"] for x in res])),
                   "mean_token_cost": float(np.mean([x["token"] for x in res])),
                   "mean_total_cost": float(np.mean([x["total"] for x in res])),
                   "mean_tokens": float(np.mean([x["n_tokens"] for x in res])),
                   "mean_latency_s": float(np.mean([x["latency_s"] for x in res])),
                   "mean_n_tools": float(np.mean([len(x["tools"]) for x in res])),
                   **deg, "wall_s": round(time.time() - t0, 1)}
            rows.append(row)
            if seed == 0:
                preds[name] = yp
                pcs[name] = per_class_f1(truth, yp, classes)
            print(f"  [{name:<12} seed={seed}] macro-F1={row['macro_f1']:.4f} "
                  f"binF1={row['binary_f1']:.4f} (base {row['binary_majority_baseline']:.4f}) "
                  f"cost={row['mean_total_cost']:.2f} tok={row['mean_tokens']:.0f} "
                  f"dead={deg['n_dead_classes']}")

    # epsilon sweep
    if a.epsilon_sweep:
        print("\n" + "-" * 70)
        print("EPSILON SWEEP — the accuracy floor is the control parameter")
        print("-" * 70)
        sweep = []
        for eps in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
            c = select_core_by_tier(evaluate_subset, all_tools, costs, eps,
                                    verbose=False)
            c = prune_harmful_core(c, evaluate_subset, verbose=False)
            v = measure_marginal_values(evaluate_subset, all_tools, c, verbose=False)
            p = ConstrainedPolicy(c, v, costs, costs.tools(c) * 1.5, eps)
            res = [pipe.run_one(r, p.select(reg.for_group(r.get("device", ""))))
                   for r in recs]
            yp = [x["label"] for x in res]
            mf = macro_f1(truth, yp)
            ct = float(np.mean([x["total"] for x in res]))
            tiers_chosen = "+".join(sorted(t.rsplit("_clf_", 1)[-1] for t in c
                                           if "_clf_" in t))
            sweep.append({"epsilon": eps, "macro_f1": mf, "cost": ct,
                          "core": c, "tiers": tiers_chosen})
            print(f"  eps={eps:<6} macro-F1={mf:.4f}  cost={ct:>10.1f}  "
                  f"tiers={tiers_chosen}")
        pd.DataFrame([{k: v for k, v in s.items() if k != "core"}
                      for s in sweep]).to_csv(
            os.path.join(out, "epsilon_sweep.csv"), index=False)
        print(f"\n  wrote {out}/epsilon_sweep.csv")

    fuser.save_cache()
    pd.DataFrame(rows).to_csv(os.path.join(out, "main_results_raw.csv"), index=False)
    pd.DataFrame(pcs).to_csv(os.path.join(out, "per_class_f1.csv"))
    with open(os.path.join(out, "predictions.json"), "w") as fh:
        json.dump({"predictions": preds, "truth": truth}, fh)
    with open(os.path.join(out, "policy.json"), "w") as fh:
        json.dump({**pol.describe(), "fusion": a.fusion,
                   "fuser_stats": fuser.stats(),
                   "cost_provenance": costs.provenance,
                   "tool_cost_ncu": costs.tool_cost}, fh, indent=2)
    print(f"\nwrote results -> {out}/")
    print("next:  python scripts/07_analysis.py --results", out)


if __name__ == "__main__":
    main()
