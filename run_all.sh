#!/usr/bin/env bash
# ----------------------------------------
# run_all.sh — the whole project, start to finish, with checks between stages.
#
#   ./run_all.sh smoke              synthetic sanity check, 2 min, no downloads
#   ./run_all.sh ton                TON_IoT, deterministic         (~1.5 h)
#   ./run_all.sh ton-llm            TON_IoT, LLM fusion            (costs money)
#   ./run_all.sh bot                Bot-IoT, deterministic         (~2 h)
#   ./run_all.sh telemetry          OPTIONAL extra corpus; needs 7 more files
#   ./run_all.sh figures ton        regenerate figures for a corpus
#
# Every stage stops on failure. Verification gates fail loudly rather than
# letting a broken configuration run for an hour.
# ----------------------------------------
set -euo pipefail

PY="${PY:-python}"
SEEDS="${SEEDS:-5}"
LIMIT="${LIMIT:-12000}"
LLM_LIMIT="${LLM_LIMIT:-1200}"
LLM_SEEDS="${LLM_SEEDS:-3}"

say()  { printf "\n\033[1m>> %s\033[0m\n" "$*"; }
ok()   { printf "  \033[32mOK\033[0m  %s\n" "$*"; }
die()  { printf "\n  \033[31mSTOP\033[0m  %s\n\n" "$*"; exit 1; }

# ---------------------------------------------------------------- gates
check_groups() {   # $1 = data dir
  $PY - "$1" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(f"{sys.argv[1]}/prepared.csv", usecols=["device"])
g = d.device.value_counts()
print("  groups:", dict(g))
tiny = [k for k, v in g.items() if v < 5000 and not k.endswith("_other")]
if tiny:
    sys.exit(f"groups too small for their own detectors: {tiny}. "
             "Raise --min-group-rows in step 01.")
if len(g) < 2:
    sys.exit("only one group; routing is doing nothing.")
EOF
}

check_costs() {    # $1 = data dir
  $PY - "$1" <<'EOF'
import json, sys
c = json.load(open(f"{sys.argv[1]}/measured_costs.json"))
if c.get("provenance") != "measured":
    sys.exit("costs are not measured.")
tc = c["tool_cost_ncu"]
groups = {k.rsplit("_clf_", 1)[0] for k in tc if "_clf_" in k}
bad = []
for g in groups:
    v = [round(tc[f"{g}_clf_{t}"], 3) for t in ("lite", "mid", "full")
         if f"{g}_clf_{t}" in tc]
    print(f"  {g:<14} tiers: {v}")
    if len(set(v)) <= 1:
        bad.append(g)
if bad:
    sys.exit(f"tiers measured identically for {bad}. The selection problem is "
             "vacuous; check the batched timing path in 03_measure_costs.py.")
EOF
}

check_results() {  # $1 = results dir
  $PY - "$1" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(f"{sys.argv[1]}/main_results_raw.csv")
g = d.groupby("condition")[["macro_f1", "mean_total_cost",
                            "n_dead_classes"]].mean()
print(g.round(4).to_string())
if "no_core" in g.index and "core_only" in g.index:
    if g.loc["no_core", "macro_f1"] > 0.6 * g.loc["core_only", "macro_f1"]:
        sys.exit("no_core did not collapse. Its budget is too generous and the "
                 "ablation proves nothing.")
    print("\n  ablation: no_core collapsed as intended")
EOF
}

# ---------------------------------------------------------------- stages
stage_prepare_ton() {
  say "01  prepare TON_IoT"
  [ -f raw/train_test_network.csv ] || die "raw/train_test_network.csv not found. See RUN.md."
  $PY scripts/01_prepare.py --corpus ton_iot \
      --input raw/train_test_network.csv --out data/ton_iot/
  check_groups data/ton_iot; ok "three viable groups"
}

stage_prepare_telem() {
  say "01  prepare TON_IoT telemetry (OPTIONAL — not needed for the paper)"
  [ -d raw/ton_telemetry ] || die "raw/ton_telemetry/ not found. See RUN.md."
  $PY scripts/01_prepare.py --corpus ton_telemetry \
      --input raw/ton_telemetry/ --out data/ton_telemetry/
  check_groups data/ton_telemetry; ok "viable device groups"
}

stage_prepare_bot() {
  say "01  prepare Bot-IoT"
  [ -d raw/botiot_full ] || die "raw/botiot_full/ not found. See RUN.md."
  $PY scripts/01_prepare.py --corpus bot_iot \
      --input raw/botiot_full/ --out data/bot_iot/ --drop-small
  check_groups data/bot_iot; ok "viable groups"
}

stage_core() {     # $1 = data dir
  say "02  build evaluation set"
  $PY scripts/02_build_eval_set.py --data-dir "$1" --allow-unequal

  say "03  MEASURE tool costs"
  $PY scripts/03_measure_costs.py --data-dir "$1" --n-incidents 200 --repeats 5
  check_costs "$1"; ok "tiers differentiate"

  say "04  achievable ceiling"
  $PY scripts/04_ceiling_check.py --data-dir "$1"
}

stage_run() {      # $1 = data dir
  say "05  conditions (deterministic) + epsilon sweep"
  $PY scripts/05_run_conditions.py --data-dir "$1" \
      --seeds "$SEEDS" --limit "$LIMIT" --epsilon-sweep
  check_results "$1/results"; ok "results sane"

  say "06  prompt injection, both payload sets"
  $PY scripts/06_injection.py --data-dir "$1" --n 300
  $PY scripts/06_injection.py --data-dir "$1" --n 300 --held-out

  say "07  tables and paired statistics"
  $PY scripts/07_analysis.py --results "$1/results"
}

stage_figures() {  # $1 = data dir, $2 = optional comparison corpus
  say "08  figures"
  CMP=""
  [ -n "${2:-}" ] && [ -f "$2/ceiling.json" ] && CMP="--compare-dir $2"
  [ -z "$CMP" ] && printf "  note: no comparison corpus; fig7 and fig10 will be\n  skipped rather than drawn from remembered numbers.\n"
  $PY scripts/08_figures.py --results "$1/results" \
      --out paper_figures/ --data-dir "$1" $CMP
  say "09  overlap audit"
  $PY scripts/09_audit_figures.py scripts/08_figures.py "$1/results" "$1" "${2:-}"
}

# ---------------------------------------------------------------- main
case "${1:-}" in
  smoke)
    say "SMOKE TEST — synthetic data, results are meaningless by design"
    $PY scripts/00_make_synthetic.py --out data/smoke --n 9000
    $PY scripts/02_build_eval_set.py --data-dir data/smoke
    $PY scripts/03_measure_costs.py --data-dir data/smoke --n-incidents 60 --repeats 3
    $PY scripts/04_ceiling_check.py --data-dir data/smoke
    $PY scripts/05_run_conditions.py --data-dir data/smoke --seeds 1 --limit 300
    check_results data/smoke/results
    $PY scripts/07_analysis.py --results data/smoke/results
    say "SMOKE TEST PASSED — the machinery works. Now use real data."
    ;;
  ton)
    stage_prepare_ton; stage_core data/ton_iot; stage_run data/ton_iot
    stage_figures data/ton_iot data/bot_iot
    say "TON_IoT COMPLETE — results in data/ton_iot/results/"
    ;;
  telemetry)
    printf "\n  OPTIONAL corpus. The paper does not need it: fig7 and fig10\n"
    printf "  compare TON_IoT against Bot-IoT, both of which you already have.\n"
    printf "  Only run this if you have the 7 Train_Test_IoT_*.csv device files\n"
    printf "  in raw/ton_telemetry/.\n\n"
    stage_prepare_telem; stage_core data/ton_telemetry
    stage_run data/ton_telemetry
    say "TELEMETRY COMPLETE — optional third corpus"
    ;;
  bot)
    stage_prepare_bot; stage_core data/bot_iot; stage_run data/bot_iot
    stage_figures data/bot_iot data/ton_iot
    say "BOT-IOT COMPLETE — results in data/bot_iot/results/"
    ;;
  ton-llm)
    [ -n "${ANTHROPIC_API_KEY:-}" ] || die "ANTHROPIC_API_KEY is not set."
    [ -f data/ton_iot/measured_costs.json ] || die "run './run_all.sh ton' first."
    say "03b  price tokens in the same unit as tool compute"
    $PY scripts/03_measure_costs.py --data-dir data/ton_iot/ \
        --n-incidents 200 --repeats 5 --time-llm
    say "05b  conditions with real LLM fusion"
    $PY scripts/05_run_conditions.py --data-dir data/ton_iot/ \
        --fusion llm --seeds "$LLM_SEEDS" --limit "$LLM_LIMIT" \
        --out data/ton_iot/results_llm
    check_results data/ton_iot/results_llm
    $PY scripts/07_analysis.py --results data/ton_iot/results_llm
    say "LLM RUN COMPLETE — cached in data/ton_iot/results_llm/llm_cache.json"
    ;;
  figures)
    stage_figures "data/${2:-ton_iot}" "data/${3:-bot_iot}"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
