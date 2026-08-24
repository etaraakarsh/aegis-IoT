# Results

## ton_iot and bot_iot

The results reported in the paper, exactly as produced by `run_all.sh`.

| File | Contents |
|---|---|
| `main_results_raw.csv` | per condition, per seed: macro-F1, binary F1, costs, dead classes |
| `per_class_f1.csv` | per-class F1 for every condition |
| `paired_tests.csv` | bootstrap paired comparisons against exhaustive invocation |
| `epsilon_sweep.csv` | the cost-accuracy frontier traced by the accuracy floor |
| `policy.json` | mandatory core, marginal values, tool costs, fusion mode |
| `ceiling.json` | achievable macro-F1, pooled and averaged, plus binary |
| `measured_costs.json` | timed tool costs, fixed overhead, and the hardware |
| `split_meta.json` | partition sizes, benign fraction, majority-class baselines |
| `in_dist/`, `held_out/` | prompt injection summaries for both payload families |

`policy.json` records `"fusion": "deterministic"` and zero API calls for both
corpora, which is the machine-readable statement of the scope described in
Section 3.6 of the paper.

## historical

Runs v3, v4 and v5 from the earlier phase of this project, kept for provenance.

These used a different corpus, an evaluation set of 189 incidents rather than
tens of thousands, a tool registry organised by device type rather than by
network service, and fusion mediated by a local language model served through
Ollama. They record non-zero token counts for that reason.

They are the runs in which the failure was first observed: in `v5`, the
condition named `aegis` scores 0.0786 macro-F1 against 0.6659 for `all`, and
`tool_usage.csv` shows the three classifier columns at exactly zero, meaning the
policy declined to call them on any incident. That is the collapse described in
the Introduction of the paper.

They are not comparable to the results above and support no claim in the paper
beyond the existence of the failure that motivated it.
