# use the venv (sentence-transformers) when it exists, else bare python3 + the hashed backend
PYBIN := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
PY := PYTHONPATH=. $(PYBIN)
DOCS ?= 1200
QUERIES ?= 1200
ROWS ?= 1250
LANGS ?= hi,ta,bn
CLIPS ?= data/audio
PORT ?= 8000
N ?= 300

.PHONY: all corpus chunking calibrate latency guardrails demo serve tune llm-probe stt legs ann verify-tune submit check venv clean

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

## exact vs HNSW: what the approximation costs, and the corpus size where it starts paying.
## EMBEDDER=hash keeps torch out of the process so faiss can build on every core -- the
## vectors are read from disk, nothing is embedded, so the backend is irrelevant here.
ann:
	EMBEDDER=hash $(PY) d1/ann.py

## gate 4, both verifiers, on the same 280 rows -> reports/verify.md
verify-tune:
	$(PY) d4/verify_tune.py

## ask one question through the serving path and watch the clock
## make demo Q="..."   |   make demo AUDIO=data/audio/clip.wav
demo:
	$(PY) service/ask.py $(if $(AUDIO),--audio $(AUDIO),) $(Q)

## does a generator fit the 200 ms budget? measured over PROBE_N calls
## make llm-probe LLM_PROVIDER=groq   |   make llm-probe LLM_PROVIDER=xai
llm-probe:
	$(PY) service/llm.py --probe

## sweep the answer-path variants against MS MARCO's own answers (make tune N=1200)
tune:
	N=$(N) $(PY) service/tune.py $(N)

## the browser demo: page and API on one origin, index loaded before the socket opens
serve:
	PORT=$(PORT) $(PY) service/server.py

## both excluded legs: real Sarvam round trips (STT over CLIPS, TTS over answer-shaped
## text). Needs SARVAM_API_KEY; writes data/excluded_legs.json for D3 to publish.
legs stt:
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
	$(PY) service/server.py --selfcheck
	$(PY) service/calibrate.py --selfcheck
	$(PY) service/tune.py --selfcheck
	$(PY) service/llm.py --selfcheck
	$(PY) d1/ann.py --selfcheck
	$(PY) d4/verify_tune.py --selfcheck
	$(PY) d1/ingest.py --selfcheck
	$(PY) stt/sarvam.py
	$(PY) stt/measure.py --selfcheck
	$(PY) tts/sarvam.py

## install the real embedder (sentence-transformers + multilingual-e5-small)
venv:
	uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python sentence-transformers

clean:
	rm -rf artifacts reports/chunking.* reports/latency.* reports/guardrails.md \
	       reports/human_sample_50.jsonl
