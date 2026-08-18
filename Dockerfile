# The serving path, containerised. Works unchanged on Hugging Face Spaces (Docker SDK),
# Fly.io and Render -- the only host-specific thing is PORT, which the server already reads.
#
# THE TWO DECISIONS THAT MATTER HERE
#   1. torch comes from the CPU wheel index. The default wheel bundles CUDA: ~2.5 GB of
#      driver that a CPU box will never call, and on a free tier it is the difference
#      between an image that builds and one that runs out of disk.
#   2. both models are downloaded AT BUILD TIME, into the image. service/server.py loads
#      the index and warms the encoder before it opens the socket, on purpose -- a request
#      that pays a 10 s model load is not a request the 200 ms budget describes. Downloading
#      600 MB on the first request would publish that lie to whoever tries it first.
FROM python:3.13-slim

# curl is for HEALTHCHECK; libgomp1 is torch's OpenMP runtime and is not in -slim
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000 and mounts a writable /home/user. Matching that here
# means the same image runs identically on Spaces and on a plain `docker run`.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

# CPU torch first and on its own layer: it is the largest and least-changing dependency,
# so it stays cached while the app churns.
RUN pip install --no-cache-dir --user \
      --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the weights in. e5-small is the query/passage encoder; the cross-encoder is gate-3.5's
# reranker, which this repo measured as worth its 26 ms.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-small'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', max_length=384); \
print('models cached')"

# The index is gitignored (it is a build artifact, 29 MB), so it must be present in the
# build context. `make chunking` produces it; DEPLOY.md says so rather than letting the
# COPY fail with a message nobody can act on.
COPY --chown=user artifacts/d1/s7 ./artifacts/d1/s7
# reports/chunking.json is not documentation here, it is configuration: winner_dir() reads
# the winning strategy out of it, and without the file it silently falls back to s2 -- an
# index this image does not carry, so the server would exit at boot naming a directory that
# was never meant to be there. The rest of reports/ rides along as the provenance for every
# number the page and the README quote.
COPY --chown=user reports ./reports
COPY --chown=user data ./data
COPY --chown=user web ./web
COPY --chown=user harness ./harness
COPY --chown=user d1 ./d1
COPY --chown=user d3 ./d3
COPY --chown=user d4 ./d4
COPY --chown=user service ./service
COPY --chown=user stt ./stt
COPY --chown=user tts ./tts

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:${PORT}/health || exit 1
CMD ["python", "service/server.py"]
