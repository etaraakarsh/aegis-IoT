#!/usr/bin/env python3
"""
03_measure_costs.py — MEASURE tool costs. Run this before any cost result.

This script exists because the previous version of this project shipped
hand-written placeholder costs into a manuscript draft. The headline "5.8x
cheaper" was arithmetic on constants that had been invented, with a code
comment reading "REPLACE with measured values before publishing" that was
never acted on.

What happens here:

  1. Fit the full tool registry on the training split.
  2. Time every tool TWICE, because a single number hides what matters:

       MARGINAL compute  — batched inference, divided by batch size. This is
                           the work the tool actually does, and it is what
                           distinguishes a 60-tree lite classifier from a
                           300-tree full one.
       FIXED overhead    — single-row call minus marginal. Framework
                           validation, array construction, dispatch. Paid once
                           per invocation regardless of which tool you chose.

     Measuring only single-row calls conflates the two, and on fast hardware
     the fixed term swamps everything: an earlier run on Apple Silicon returned
     14.15 ms for BOTH the lite and full classifiers, making the tiers
     indistinguishable and the whole selection problem vacuous.

     NCU is based on marginal compute, since that is the term the policy's
     decision actually depends on. The fixed overhead is reported separately
     and belongs in your Methods section.
  3. Optionally time the LLM fusion path to price tokens in the same unit.
  4. Normalize so the cheapest measured tool == 1.0 NCU and write
     data/measured_costs.json.

Report the hardware in your Methods section. These numbers are properties of
your machine, and a reader on different hardware should be told so.

    python scripts/03_measure_costs.py --data-dir data/ton_iot/
    python scripts/03_measure_costs.py --data-dir data/ton_iot/ --time-llm
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aegis.costs import save_costs
from aegis.tools import Registry

WARMUP = 20
REPEATS = 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-incidents", type=int, default=200,
                    help="incidents per timing pass")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--time-llm", action="store_true",
                    help="also time real LLM fusion to price tokens (costs money)")
    ap.add_argument("--llm-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    train = pd.read_csv(os.path.join(a.data_dir, "train.csv"), low_memory=False)
    val = pd.read_csv(os.path.join(a.data_dir, "val.csv"), low_memory=False)
    meta = ["true_type", "device", "domain"]
    feats = [c for c in train.columns if c not in meta]

    print(f"\n=== measuring tool costs on {platform.processor() or platform.machine()} ===")
    print(f"  python {platform.python_version()}  |  {platform.system()} "
          f"{platform.release()}")
    reg = Registry(seed=a.seed).fit(train, feats)

    sample = val.sample(min(a.n_incidents, len(val)),
                        random_state=a.seed).to_dict("records")
    by_group = {}
    for r in sample:
        by_group.setdefault(r.get("device", ""), []).append(r)

    print(f"\n  timing {len(reg.tools)} tools, {a.repeats} passes of "
          f"{a.n_incidents} incidents (median reported)")
    print(f"  {'tool':<28}{'sec/call':>12}{'rel':>9}")
    print("  " + "-" * 50)

    seconds, single_call = {}, {}
    for name in reg.names():
        tool = reg.tools[name]
        rows = by_group.get(tool.group, [])
        if not rows:
            continue

        # single-call path (marginal + fixed overhead)
        for r in rows[:WARMUP]:
            reg.invoke(name, r)
        passes = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            for r in rows:
                reg.invoke(name, r)
            passes.append((time.perf_counter() - t0) / len(rows))
        single_call[name] = statistics.median(passes)

        # batched path (marginal compute only)
        X = np.array([[float(r.get(c, 0.0)) for c in tool.features]
                      for r in rows], dtype=float)
        def _batch():
            if tool.kind == "classifier":
                tool.predict_proba(X)
            elif tool.kind == "isoforest":
                tool.score(X)
            else:                        # rules / drift are pure numpy
                lo_hi = tool.model
                if tool.kind == "rules":
                    lo, hi = lo_hi
                    ((X < lo) | (X > hi)).mean(axis=1)
                else:
                    mu, sd = lo_hi
                    np.abs((X - mu) / sd).mean(axis=1)
        _batch()
        passes = []
        for _ in range(a.repeats):
            t0 = time.perf_counter()
            _batch()
            passes.append((time.perf_counter() - t0) / len(rows))
        seconds[name] = max(statistics.median(passes), 1e-9)

    floor = min(v for v in seconds.values() if v > 0)
    overheads = [single_call[n] - seconds[n] for n in seconds]
    med_overhead = statistics.median(overheads) if overheads else 0.0
    print(f"  {'tool':<28}{'marginal':>12}{'single':>12}{'rel':>9}")
    print("  " + "-" * 62)
    for name in sorted(seconds, key=lambda k: -seconds[k]):
        print(f"  {name:<28}{seconds[name]:>12.3e}{single_call[name]:>12.3e}"
              f"{seconds[name]/floor:>9.2f}")
    print(f"\n  fixed per-invocation overhead (median): {med_overhead:.3e} s")
    print( "  NCU is based on MARGINAL compute. Report the overhead in Methods;")
    print( "  it is paid once per call regardless of which tool is chosen.")

    tier_costs = {t: seconds[n] for n in seconds
                  for t in [reg.tools[n].tier] if reg.tools[n].tier}
    if len({round(v, 12) for v in tier_costs.values()}) <= 1 and tier_costs:
        print("\n  WARNING: classifier tiers measured identically. The tiered")
        print("  registry cannot be exercised; check the batch path.")

    # token pricing
    token_sec_per_1k = 0.0
    if a.time_llm:
        from aegis.llm import Fuser
        print(f"\n  timing LLM fusion on {a.llm_samples} incidents "
              f"(this makes real API calls)")
        fu = Fuser(mode="llm")
        classes = sorted(train.true_type.unique())
        tot_s, tot_t = 0.0, 0
        for r in sample[:a.llm_samples]:
            ev = [e for e in (reg.invoke(t, r)
                              for t in reg.for_group(r.get("device", "")))
                  if e]
            res = fu.fuse(ev, classes)
            tot_s += res.latency_s
            tot_t += res.n_tokens
        if tot_t:
            token_sec_per_1k = tot_s / (tot_t / 1000.0)
            print(f"  {tot_t} tokens in {tot_s:.2f}s "
                  f"-> {token_sec_per_1k:.4f} s per 1k tokens "
                  f"= {token_sec_per_1k/floor:.2f} NCU/1k")
    else:
        print("\n  token cost NOT measured (no --time-llm).")
        print("  Cost results will cover TOOL COMPUTE ONLY. Say so in the paper,")
        print("  or re-run with --time-llm before making an efficiency claim.")

    path = save_costs(a.data_dir, seconds, token_sec_per_1k, {
        "fixed_overhead_seconds_per_call": med_overhead,
        "single_call_seconds": {k: round(v, 9) for k, v in single_call.items()},
        "cost_basis": "marginal batched compute per incident",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "n_incidents": a.n_incidents,
        "repeats": a.repeats,
        "statistic": "median over repeats of batched seconds per incident",
        "token_cost_measured": bool(a.time_llm and token_sec_per_1k > 0),
    })
    print(f"\nwrote {path}")
    print("\nQuote the hardware in your Methods section. These are properties")
    print("of this machine, and a reader on other hardware must be told.")
    print("\nnext:  python scripts/04_ceiling_check.py --data-dir", a.data_dir)


if __name__ == "__main__":
    main()
