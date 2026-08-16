# use the venv (sentence-transformers) when it exists, else bare python3 + the hashed backend
PYBIN := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PY := PYTHONPATH=. $(PYBIN)
DOCS ?= 1200
QUERIES ?= 1200
ROWS ?= 1250
LANGS ?= hi,ta,bn
CLIPS ?= data/audio

.PHONY: all corpus chunking calibrate latency guardrails demo stt submit check venv clean

all: submit

## stage 0 -- stream the real corpus (ai4bharat/MSMARCO-XI) into data/raw/
corpus:
	$(PY) d1/ingest.py --langs $(LANGS) --rows $(ROWS)

## D1 -- 8 chunking strategies x 6 columns, offline
chunking:
	$(PY) d1/run.py --docs $(DOCS) --queries $(QUERIES)

## gate 2's dense-score floor, measured on D3's frozen set (needs `make chunking` first)
calibrate:
	$(PY) service/calibrate.py

## D3 -- record 500 queries cold/warm/concurrent, then reduce to tables + chart
latency:
	$(PY) d3/run.py
	$(PY) d3/reduce.py

## D4 -- 280 labelled queries, both error directions
guardrails:
	$(PY) d4/run.py

## ask one question through the serving path and watch the clock
## make demo Q="..."   |   make demo AUDIO=data/audio/clip.wav
demo:
	$(PY) service/ask.py $(if $(AUDIO),--audio $(AUDIO),) $(Q)

## the excluded legs: real Sarvam round trips over CLIPS, published beside the budget
stt:
	CLIPS=$(CLIPS) $(PY) stt/measure.py

## all three, then splice the tables into README.md. calibrate sits between D1 and the two
## reports that are served through gate 2, so they are graded on a floor fitted to this corpus.
submit: chunking calibrate guardrails latency
	$(PY) harness/splice.py

## every self-check in the repo -- the runnable proof the logic still holds.
## EMBEDDER=hash: the checks test logic, not the model, and stay dependency-free and fast.
check: export EMBEDDER=hash
check:
	$(PY) harness/spans.py
	$(PY) harness/budget.py
	$(PY) harness/env.py > /dev/null && echo "env ok"
	$(PY) harness/splice.py --selfcheck
	$(PY) d1/corpus.py
	$(PY) d1/index.py
	$(PY) d1/chunkers.py
	$(PY) d1/run.py --selfcheck
	$(PY) d3/run.py --selfcheck
	$(PY) d3/reduce.py --selfcheck
	$(PY) d4/dataset.py
	$(PY) d4/run.py --selfcheck
	$(PY) service/pipeline.py
	$(PY) service/ask.py --selfcheck
	$(PY) service/calibrate.py --selfcheck
	$(PY) d1/ingest.py --selfcheck
	$(PY) stt/sarvam.py
	$(PY) stt/measure.py --selfcheck

## install the real embedder (sentence-transformers + multilingual-e5-small)
venv:
	uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python sentence-transformers

clean:
	rm -rf artifacts reports/chunking.* reports/latency.* reports/guardrails.md \
	       reports/human_sample_50.jsonl
