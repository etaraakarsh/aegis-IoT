# AEGIS-IoT v6 — DATA defaults to data/ton_iot; override on the command line.
DATA ?= data/ton_iot
SEEDS ?= 5
PY ?= python

.PHONY: smoke ton botiot costs ceiling run runllm inject analyse figures all clean

smoke:
	$(PY) scripts/00_make_synthetic.py --out data/smoke --n 9000
	$(PY) scripts/02_build_eval_set.py --data-dir data/smoke
	$(PY) scripts/03_measure_costs.py  --data-dir data/smoke --n-incidents 60 --repeats 3
	$(PY) scripts/04_ceiling_check.py  --data-dir data/smoke
	$(PY) scripts/05_run_conditions.py --data-dir data/smoke --seeds 1 --limit 300
	$(PY) scripts/07_analysis.py       --results data/smoke/results

ton:
	$(PY) scripts/01_prepare.py --corpus ton_iot --input $(INPUT) --out data/ton_iot/

botiot:
	$(PY) scripts/01_prepare.py --corpus bot_iot --input $(INPUT) --out data/bot_iot/ --drop-small

costs:   ; $(PY) scripts/03_measure_costs.py  --data-dir $(DATA)/
ceiling: ; $(PY) scripts/04_ceiling_check.py  --data-dir $(DATA)/
run:     ; $(PY) scripts/05_run_conditions.py --data-dir $(DATA)/ --seeds $(SEEDS)
runllm:  ; $(PY) scripts/05_run_conditions.py --data-dir $(DATA)/ --fusion llm --seeds 3 --limit 1500 --out $(DATA)/results_llm

inject:
	$(PY) scripts/06_injection.py --data-dir $(DATA)/ --n 300
	$(PY) scripts/06_injection.py --data-dir $(DATA)/ --n 300 --held-out

analyse: ; $(PY) scripts/07_analysis.py --results $(DATA)/results

figures:
	$(PY) scripts/08_figures.py --results $(DATA)/results --out paper_figures/ --data-dir $(DATA)/
	$(PY) scripts/09_audit_figures.py scripts/08_figures.py $(DATA)/results

all: costs ceiling run inject analyse figures

clean: ; rm -rf data/smoke paper_figures
