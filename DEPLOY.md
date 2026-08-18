# Deploying the live site

One container serves the page and the API on **one origin**, which is why there is no CORS
configuration anywhere in this repo and nothing to paste into the page. `Dockerfile` runs
unchanged on Hugging Face Spaces, Fly.io and Render — the only host-specific thing is `PORT`,
which `service/server.py` already reads.

## The one thing that is easy to get wrong

**The index is a build artifact and `artifacts/` is gitignored.** It is 29 MB, it is not in
the GitHub repo, and the image copies it from the build context. So before you deploy:

```bash
make venv && make corpus && make chunking     # ~25 min, produces artifacts/d1/s7
ls artifacts/d1/s7/index.npy                  # must exist, ~18 MB
```

`reports/chunking.json` ships too, and is configuration rather than documentation:
`winner_dir()` reads the winning strategy out of it. Without it the server silently falls
back to `artifacts/d1/s2`, an index the image does not carry, and exits at boot naming a
directory nobody put there.

## Hugging Face Spaces

Free, no card, 16 GB RAM, and it sleeps after long idleness — a cold visitor waits about
30 seconds for the models to load, and the page shows a *waking up* state while `/health`
still says `loading`. That wait is model loading, not the pipeline: the 200 ms the reports
measure is `t0 → t1` inside a request, and every request the page shows you is measured after
the box is warm.

```bash
# 1. create a Space at huggingface.co/new-space  ->  SDK: Docker, blank template
# 2. clone it and copy this project in
git clone https://huggingface.co/spaces/<you>/mic-rag && cd mic-rag
rsync -a --exclude .git /path/to/"Mic Rag model"/ .

# 3. the Space card must be the README at the Space root
cp deploy/SPACE_README.md README.md

# 4. artifacts/ is gitignored here, so add it explicitly; HF puts big files in LFS itself
git lfs install
git add -f artifacts/d1/s7 reports data web
git add Dockerfile requirements.txt .dockerignore README.md service stt tts harness d1 d3 d4
git commit -m "Mic RAG: page and API on one origin"
git push
```

Then in **Settings → Variables and secrets**, add `SARVAM_API_KEY`. Without it the box still
runs and typed questions work end to end; the microphone returns a 502 naming the missing
key, because speech-to-text is the one part of this system that is somebody else's service.

Build takes about 10 minutes, most of it the CPU torch wheel and the 600 MB of model weights
that are baked into the image on purpose — a request that pays a model load is not a request
the 200 ms budget describes.

## Fly.io or Render

Same image, no sleeping, roughly $7–10/month. This is what the README's deploy note
recommends, and it is the right call if the latency story has to hold for every visitor
including the first.

```bash
fly launch --dockerfile Dockerfile --vm-memory 4096 --no-deploy
fly secrets set SARVAM_API_KEY=...
fly deploy
```

Give it **4 GB**. Two transformer models plus the index plus torch will survive on 2 GB and
will start swapping under any concurrency, which turns a latency demo into a latency
counter-example.

## Locally

```bash
make serve                      # http://localhost:8000
docker build -t mic-rag . && docker run -p 7860:7860 -e SARVAM_API_KEY=... mic-rag
```

## What to check once it is up

```bash
curl -s https://<your-host>/health          # {"status":"ok","index":"artifacts/d1/s7",...}
curl -s -X POST https://<your-host>/ask \
     -H 'Content-Type: application/json' -d '{"text":"what is a corporation"}'
```

`status: ok` means the index is loaded and the encoder is warm — the server opens its socket
only after both, so a box that answers `/health` is a box whose first real request is already
inside the budget. If `/ask` abstains with `gate2_score`, that is the system working: the
corpus is 1200 MS MARCO passages and most of the world is not in it.
