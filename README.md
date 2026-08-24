# Accuracy-Constrained Tool Selection for Multi-Component Intrusion Detection in IoT Networks

Code, data preparation, results and figures for the paper:

> **When Cost Objectives Delete Capability: Accuracy-Constrained Tool Selection
> for Multi-Component Intrusion Detection in IoT Networks**
> Aakarsh Etar, Jayesh Soni, Himanshu Upadhyay
> *Future Internet* (MDPI), manuscript futureinternet-4524832

## What this repository shows

A tool-invocation policy trained against cost alone removes the detectors that
carry class-discriminative signal, because in a diagnostic pipeline the
informative tools are usually the expensive ones. We document that failure,
propose a constrained formulation that prevents it, and measure both on two
public corpora with timed invocation costs.

Every number in the paper can be regenerated from this repository. The results
we submitted are included under `results/`, so a reader can inspect them without
downloading the corpora or re-running anything.

## Headline results

TON_IoT network, 5 seeds, 12,000 test incidents. Measured ceiling 0.9578,
majority-class binary F1 0.7876.

| Condition | macro-F1 | Cost (NCU) | vs exhaustive | Dead classes |
|---|---|---|---|---|
| Core removed | 0.0511 | 2.0 | — | 8 of 10 |
| Constrained (eps=0.02) | 0.9542 | 1120 | 3.08x cheaper | 0 |
| Fixed | 0.9555 | 2542 | 1.36x | 0 |
| Exhaustive | 0.9549 | 3451 | 1.0x | 0 |

Bot-IoT, 5 seeds, 8,181 test incidents. Measured ceiling 0.9999, majority-class
binary F1 0.7839.

| Condition | macro-F1 | Cost (NCU) | vs exhaustive | Dead classes |
|---|---|---|---|---|
| Core removed | 0.2422 | 1.0 | — | 2 of 4 |
| Constrained | 0.9997 | 294 | 6.98x cheaper | 0 |
| Fixed | 0.9999 | 1469 | 1.40x | 0 |
| Exhaustive | 0.9990 | 2051 | 1.0x | 0 |

The accuracy floor traces a cost-accuracy frontier on TON_IoT:

| eps | Tiers selected | macro-F1 | Cost (NCU) | vs exhaustive |
|---|---|---|---|---|
| 0.005 | full + lite + mid | 0.9534 | 1536 | 2.25x |
| 0.02 | lite + mid + mid | 0.9542 | 1120 | 3.08x |
| 0.05 | lite + lite + mid | 0.9411 | 768 | 4.49x |
| 0.20 | lite + lite + lite | 0.9360 | 482 | 7.16x |

Bot-IoT cannot exercise this frontier: with a ceiling of 0.9999 every capacity
tier scores identically on validation, so the cheapest satisfies any tolerance.

## Scope

The pipeline evaluated in the paper fuses evidence deterministically. No
language model participates in the reported experiments, and token cost is zero
by construction rather than by measurement. The cost figures therefore
characterise the tool-selection layer.

The failure that motivated this work was first observed in an earlier
configuration where fusion was mediated by a local language model served
through Ollama. Those runs are preserved under `results/historical/` for
provenance. They use a different corpus, a much smaller evaluation set, and a
different tool registry, so they are not comparable to the results in the paper
and are not used to support any claim in it.

## Repository layout

    aegis/          library: corpora, tool registry, cost model, policy,
                    pipeline, fusion backends
    scripts/        numbered pipeline stages, 00 through 09
    data/           where to place the downloaded corpora (see data/README.md)
    results/        the results reported in the paper, plus historical runs
    figures/        the figures as they appear in the paper
    docs/           full run guide
    run_all.sh      runs the whole pipeline end to end

## Quick start

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    chmod +x run_all.sh
    ./run_all.sh smoke

The smoke test uses synthetic data, takes about two minutes and needs no
downloads. It confirms the pipeline runs and that the core-removal condition
collapses as intended.

For the real corpora see `data/README.md`, then:

    ./run_all.sh ton
    ./run_all.sh bot

## Reproducing the paper

Timed costs are properties of the machine that measured them. Ours were taken
on Apple Silicon under macOS 15.5 with Python 3.14.6, and are recorded in
`results/*/measured_costs.json` along with the hardware. Absolute cost figures
will differ on other hardware; the ratios between capacity tiers, roughly
1 : 2.4 : 4.9, held across both corpora and are the transferable quantity.

Accuracy figures are deterministic given the corpus and the seed, and should
reproduce exactly.

## License

MIT. See `LICENSE`.
