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
    # weights into the image, not into the first request. service/server.py loads the index
    # and warms the encoder before it opens the socket, which is the whole reason its first
    # answer is already inside the budget; downloading at runtime would undo that.
    .run_commands(
        "python -c \""
        "from sentence_transformers import SentenceTransformer, CrossEncoder; "
        "SentenceTransformer('intfloat/multilingual-e5-small'); "
        "CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', max_length=384)\""
    )
    .env({"PYTHONPATH": APP_DIR, "PYTHONUNBUFFERED": "1", "PORT": str(PORT)})
    # only the winning strategy: the other seven indexes are 420 MB of evidence for
    # reports/chunking.md, and this box serves the one reports/chunking.json names
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
    .add_local_file("data/corpus.jsonl", f"{APP_DIR}/data/corpus.jsonl")
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
    cpu=2,                    # the cross-encoder runs two lanes; one core makes them queue
    memory=4096,              # two transformers + the index. 2 GB swaps under concurrency,
                              # which turns a latency demo into a latency counter-example
    secrets=SECRETS,
    scaledown_window=300,     # stay warm 5 min between questions: only the first pays boot
    max_containers=1,         # one box, so the semantic cache and the warm encoder are shared
)
@modal.concurrent(max_inputs=4)   # ThreadingHTTPServer handles these itself
@modal.web_server(port=PORT, startup_timeout=600)
def serve():
    """Start the same server `make serve` starts, in the same layout."""
    subprocess.Popen(["python", "service/server.py"], cwd=APP_DIR)
