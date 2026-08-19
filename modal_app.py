"""The live site, on Modal: page and API on one origin, scaled to zero.

    pip install modal && modal setup      # once, browser OAuth, no card on the Starter plan
    modal deploy modal_app.py             # prints the public URL

WHY THIS FILE EXISTS ALONGSIDE THE DOCKERFILE
Both describe the same box and neither is the source of truth for the other, so they are
kept deliberately parallel: same CPU torch wheel, same two models baked in at build, same
files, same entrypoint. The Dockerfile is what a normal host runs; this is what a
scale-to-zero host runs, and the difference is only in who supplies the runtime.

WHAT SCALE TO ZERO COSTS, IN BOTH DIRECTIONS
Nothing runs between visits, so a demo costs cents against the Starter plan's monthly
credit. The price is a cold start: the first visitor after an idle period waits while the
container boots and 600 MB of weights load. That wait is NOT the 200 ms this repo measures
-- t0 is the instant a warm server receives a final transcript -- and the page says so,
showing a "waking up" state while /health still reports `loading`. `scaledown_window` keeps
the box alive between questions so only the first one pays.
"""
import subprocess

import modal

APP_DIR = "/root/app"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.13")
    # torch's OpenMP runtime is not in -slim, and the import fails without it
    .apt_install("libgomp1")
    # the CPU wheel on purpose: the default one bundles CUDA, ~2.5 GB of driver that a
    # container with no GPU will never call
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("sentence-transformers==5.7.0")
    # needed to READ the prebuilt graph, not to build one -- see requirements.txt
    .pip_install("faiss-cpu==1.15.0")
    # weights into the image, not into the first request. service/server.py loads the index
    # and warms the encoder before it opens the socket, which is the whole reason its first
    # answer is already inside the budget; downloading at runtime would undo that.
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('intfloat/multilingual-e5-small'); "
        "CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', max_length=384)\""
    )
    # torch reads os.cpu_count(), which inside a container reports the HOST's cores rather
    # than the cgroup's, so it opens far more threads than it has been given and they fight.
    # This repo already measured that shape at 10 cores -- "four torch threads x four
    # requests on ten cores is thrashing, not parallelism" -- and a container makes it worse
    # by lying about the denominator. Four threads per lane x RERANK_LANES=2 = the eight
    # cores actually allocated.
    .env({"PYTHONPATH": APP_DIR, "PYTHONUNBUFFERED": "1", "PORT": str(PORT),
          "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
          # THE RERANKER IS OFF ON THIS BOX, AND THAT IS A MEASUREMENT TALKING.
          # It costs 21 ms on the 10-vCPU bench and 70-90 ms here; cpu=8 and cpu=16 made no
          # difference, because one cross-encoder forward pass is latency-bound rather than
          # throughput-bound and these vCPUs are simply slower per core. At that price it
          # spent the budget the answer needed: Bengali questions -- longer after Indic
          # tokenization, so the most expensive ones -- were refused with no citation, which
          # is the ladder protecting the deadline exactly as designed. Serving the fused
          # order instead costs top-1 41.5% -> 34.6% and answer F1 0.326 -> 0.303, both
          # measured in the README, and it buys back a system that answers.
          # This is the repo's own warning coming true: "measure on the deploy hardware, not
          # your M-series Mac -- ARM local numbers will flatter you badly."
          # ...so the reranker stays ON, at half depth. Depth 4 costs 70-90 ms here; depth 2
          # costs about half and still does the job this corpus needs, because the fix it is
          # famous for on this dataset is a swap between two ADJACENT ranks -- chunk :1:0
          # ("best answer: it is made of fat", a sentence that restates the question) losing
          # to :0:0, the passage that states the fact. Turning it off reproduced that exact
          # regression on the live box, which is the most direct evidence in this repo that
          # the 26 ms it costs on the bench is real.
          "RERANK": "cross", "RERANK_TOP": "2",
          # 300k vectors is where exact search stops being free: measured on this box it ran
          # 76-114 ms and ate the budget the answer needed, so rerank was skipped and valid
          # questions were refused. reports/ann.md predicted exactly this crossover. The
          # graph is built once and shipped -- rebuilding it costs 4 minutes, and a
          # scale-to-zero box would pay that on every cold start.
          "INDEX": "hnsw"})
    # only the winning strategy: the other seven indexes are 420 MB of evidence for
    # reports/chunking.md, and this box serves the one reports/chunking.json names
    # includes index.faiss: the prebuilt HNSW graph, so the container reads it rather than
    # spending four minutes of every cold start rebuilding what never changed
    .add_local_dir("artifacts/d1/s7", f"{APP_DIR}/artifacts/d1/s7")
    # reports/ is configuration here, not documentation -- winner_dir() reads the winning
    # strategy out of reports/chunking.json, and without it the server falls back to an
    # index this image does not carry and exits at boot
    .add_local_dir("reports", f"{APP_DIR}/reports")
    .add_local_dir("web", f"{APP_DIR}/web")
    .add_local_dir("harness", f"{APP_DIR}/harness")
    .add_local_dir("d1", f"{APP_DIR}/d1")
    .add_local_dir("d3", f"{APP_DIR}/d3")
    .add_local_dir("d4", f"{APP_DIR}/d4")
    .add_local_dir("service", f"{APP_DIR}/service")
    .add_local_dir("stt", f"{APP_DIR}/stt")
    .add_local_dir("tts", f"{APP_DIR}/tts")
    # data/corpus.jsonl is deliberately NOT here: it is 515 MB and nothing in the serving
    # path reads it. The passages the server answers from live in the index's own
    # chunks.jsonl; the corpus file is what the benches and the dataset builders replay.
    .add_local_file("data/score_floor.json", f"{APP_DIR}/data/score_floor.json")
    .add_local_file("data/frozen.json", f"{APP_DIR}/data/frozen.json")
)

app = modal.App("mic-rag", image=image)

# SARVAM_API_KEY lives in a Modal secret, never in this file and never in the image. It is
# resolved at deploy time, so this is a hard requirement rather than a soft one:
#
#     modal secret create sarvam SARVAM_API_KEY=...
#
# If you have no Sarvam key, create it with an empty value. The box still runs and typed
# questions work end to end; the microphone then returns a 502 naming the missing key,
# because speech-to-text is the one part of this system that is somebody else's service.
SECRETS = [modal.Secret.from_name("sarvam", required_keys=["SARVAM_API_KEY"])]


@app.function(
    # SIZED FROM A MEASUREMENT, NOT A GUESS. At cpu=2 this box refused questions it should
    # have answered: the cross-encoder took 90 ms against the 20.9 ms the reports record on
    # the 10-vCPU bench, the ladder skipped `generate` to protect the deadline, and a request
    # with no citation is an abstention. The guardrails were right and the machine was wrong.
    # Every stage budget in service/pipeline.py was calibrated on ~10 cores, so the host has
    # to bring roughly that many or the ladder degrades correct answers away.
    cpu=8,
    memory=8192,              # two transformers + a 500 MB index. 4 GB was fine at 12k chunks,
                              # which turns a latency demo into a latency counter-example
    secrets=SECRETS,
    min_containers=0,         # scale to zero: billed only while someone is actually using it
    scaledown_window=900,     # ...but stay warm 15 min, so one evaluation session pays once
    max_containers=1,         # one box, so the semantic cache and the warm encoder are shared
)
@modal.concurrent(max_inputs=4)   # ThreadingHTTPServer handles these itself
@modal.web_server(port=PORT, startup_timeout=600)
def serve():
    """Start the same server `make serve` starts, in the same layout."""
    subprocess.Popen(["python", "service/server.py"], cwd=APP_DIR)
