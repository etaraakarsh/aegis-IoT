# AEGIS-IoT — run guide

**Two datasets. Two things to place. Three commands.**

---

## 1. Install

```bash
unzip aegis-iot-FINAL.zip && cd aegis-iot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run_all.sh
mkdir -p raw
```

## 2. Place your data

You already have both. Nothing else is needed.

```
aegis-iot/
└── raw/
    ├── train_test_network.csv       <- TON_IoT, the single file you downloaded
    └── botiot_full/                 <- the 74 Bot-IoT shards
        ├── UNSW_2018_IoT_Botnet_Dataset_1.csv
        ├── ...
        └── UNSW_2018_IoT_Botnet_Dataset_74.csv
```

Verify:

```bash
wc -l raw/train_test_network.csv     # 211044
ls raw/botiot_full/*.csv | wc -l     # 74
```

## 3. Run

```bash
./run_all.sh smoke     # 2 min, synthetic, confirms the machinery
./run_all.sh ton       # ~1.5 h, TON_IoT — the main result
./run_all.sh bot       # ~2 h, Bot-IoT — the replication
```

That is the whole project. Figures 7 and 10 compare the two corpora against
each other, so running `ton` then `bot` gives both sides real data. Redraw
them once both are done:

```bash
./run_all.sh figures ton_iot bot_iot
```

Speed knobs:

```bash
SEEDS=3 LIMIT=6000 ./run_all.sh ton     # faster first pass
SEEDS=5 LIMIT=0    ./run_all.sh ton     # full 38,575-incident test split
```

### Optional: LLM fusion (costs money)

Only after `ton` looks right.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
LLM_LIMIT=400  LLM_SEEDS=1 ./run_all.sh ton-llm    # size the spend first
LLM_LIMIT=1200 LLM_SEEDS=3 ./run_all.sh ton-llm    # then commit
```

The script prints call count, token estimate and dollar figure, then waits six
seconds so you can Ctrl-C. Responses are cached; re-analysis with
`--fusion replay` is free.

Expect LLM fusion NOT to beat deterministic fusion. The classifier has already
made a confident call and the model is re-deriving it at ~100x the cost. That
is a legitimate result — the agentic layer's value is not in fusion accuracy —
and reporting it honestly is stronger than omitting the comparison.

### Optional: a third corpus

`./run_all.sh telemetry` exists and reads the per-device
`Train_Test_IoT_*.csv` files if you ever place them in `raw/ton_telemetry/`.
**The paper does not need it.** Figures 7 and 10 compare TON_IoT against
Bot-IoT, which you already have.

---

## 4. What changed in this build

**Default epsilon is 0.02, not 0.05.** On TON_IoT, 0.05 costs 1.5% relative
macro-F1 (p<0.001) for a 4.7x cost reduction; 0.02 costs **0.07% for 3.3x**.
Justify whichever you report with `--epsilon-sweep`.

**Figures 7 and 10 read real files** — `ceiling.json` and
`injection_summary.json` from both corpora. If the comparison corpus is
missing they are SKIPPED. The previous build filled that gap with numbers
copied from an earlier experiment, which is how a figure came to report
1.00%/0.33% for injection while the run beside it had returned 0.00%/0.00%.

**Figure 11 and Table 3b are new** — per-class F1 loss against training
frequency. On TON_IoT the rarest class (mitm, n=1,043) loses about 10%
relative under the constrained policy while classes ten times larger lose
under 1%. Capacity reduction costs rare classes first; that is the operational
limit on how far the accuracy floor can be relaxed.

**Table 5 reports injection honestly.** When every arm returns zero it prints
NULL RESULT and states what you may not conclude from it.

**The figure auditor now covers all ten figures**, not six, and reports a
signature mismatch as a message rather than a TypeError. It found and fixed a
real collision in fig9 where negative marginal-value labels ran into the
y-axis tick labels — a figure that had never been checked before.

---

## 5. Gates — check these before writing

`run_all.sh` enforces the first three and stops if they fail.

1. **Three groups for TON_IoT**: `net_other`, `net_dns`, `net_http`. For
   Bot-IoT: `bot_udp` and `bot_tcp`, with `bot_icmp` folded in after
   rebalancing.
2. **Classifier tiers differ in cost.** Expect roughly `lite : mid : full =
   1 : 2.4 : 4.9`. If all classifiers print the same number the tiered
   registry is not being exercised. Costs are batched marginal compute, with
   the fixed per-invocation overhead reported separately.
3. **`no_core` collapses** — far below every other condition, with dead
   classes, and binary F1 *below* the majority baseline.
4. **`cost provenance: measured`** in the `07_analysis.py` output.
5. **Token share is 0.000 only under deterministic fusion.**

---

## 6. Where the results land

```
data/ton_iot/results/            main_results_raw.csv, per_class_f1.csv,
                                 paired_tests.csv, epsilon_sweep.csv,
                                 policy.json, predictions.json
data/ton_iot/results/in_dist/    injection_summary.json
data/ton_iot/results/held_out/   injection_summary.json
data/ton_iot/ceiling.json
data/ton_iot/measured_costs.json
data/bot_iot/...                 same layout
paper_figures/                   fig2..fig11, PNG + PDF at 600 dpi
```

## 7. Results to expect

TON_IoT, 5 seeds x 12,000 incidents. Ceiling 0.9578, majority binary F1 0.7930:

| condition | macro-F1 | cost | vs exhaustive | dead |
|---|---|---|---|---|
| no_core | 0.0511 | 2.0 | — | **8 / 10** |
| constrained (eps=0.02) | 0.9542 | 1014.0 | **3.28x** | 0 |
| constrained (eps=0.05) | 0.9411 | 709.3 | 4.70x | 0 |
| fixed | 0.9555 | 2413.9 | 1.38x | 0 |
| exhaustive | 0.9549 | 3330.1 | 1.0x | 0 |

Bot-IoT, 5 seeds x 8,181. Ceiling 0.9999, majority binary F1 0.7839:

| condition | macro-F1 | cost | vs exhaustive | dead |
|---|---|---|---|---|
| no_core | 0.2422 | 1.1 | — | 2 / 4 |
| constrained | 0.9997 | 295.3 | **6.87x** | 0 |
| fixed | 0.9999 | 1435.6 | 1.41x | 0 |
| exhaustive | 0.9990 | 2028.0 | 1.0x | 0 |

The epsilon frontier on TON_IoT:

| eps | tiers | macro-F1 | cost | vs exhaustive | rel. loss |
|---|---|---|---|---|---|
| 0.005 | full+lite+mid | 0.9534 | 1399.3 | 2.38x | 0.16% |
| **0.02** | **lite+mid+mid** | **0.9542** | **1014.0** | **3.28x** | **0.07%** |
| 0.05 | lite+lite+mid | 0.9411 | 709.3 | 4.70x | 1.45% |
| 0.20 | lite+lite+lite | 0.9360 | 467.7 | 7.12x | 1.98% |

Bot-IoT's sweep is flat (`lite+lite` at every eps) because the corpus is
saturated — ceiling 0.9999, every tier scores 1.0000 on validation. Bot-IoT
replicates the ablation and the cheapest-tier-suffices finding; it cannot
exercise the frontier. Say that rather than presenting a flat sweep as a
result.

## 8. Reporting checklist

- Majority-class baseline beside every binary number (TON_IoT 0.7930,
  Bot-IoT 0.7839).
- Measured ceiling from `ceiling.json`; read agent scores as a fraction of it.
  Averaged per-group macro-F1 is a *different statistic* from pooled and is
  not comparable to agent scores.
- Hardware from `measured_costs.json` -> `meta`, plus the fixed
  per-invocation overhead.
- Relative effect size beside every p-value.
- "Tool compute only" in the same sentence as any deterministic-fusion cost.
- Standard deviations are 0.0000 because fusion is deterministic and seeds
  affect only the bootstrap. Say so, or a reviewer assumes one seed run five
  times.
