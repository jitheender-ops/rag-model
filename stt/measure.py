"""Measure the excluded legs and publish them beside the budget.

    export SARVAM_API_KEY=...
    make stt                      # reads data/audio/*.{wav,webm,mp3}
    make stt CLIPS=/path/to/dir

The 200 ms budget starts at t0 = the server holding a final transcript, so the STT round
trip is outside it. Outside the budget is not the same as invisible: D3's report prints
this file's numbers directly under the table, and prints "not measured on this run" when
the file is absent. That line is the whole reason this script exists.

Three rules it will not bend:
  * no key, no file. A mock's 0.0 ms published as an excluded leg is a fabricated number.
  * no clips, no file. Same reason.
  * only what was actually measured is written. This repo has no TTS stage, so tts_p50 is
    absent rather than invented, and D3 prints "_" for it.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from d3.reduce import EXCLUDED, pct
from stt import sarvam

CLIPS = os.getenv("CLIPS", "data/audio")
EXTS = ("wav", "webm", "mp3", "m4a", "ogg", "flac")
MIME = {"wav": "audio/wav", "webm": "audio/webm", "mp3": "audio/mpeg",
        "m4a": "audio/mp4", "ogg": "audio/ogg", "flac": "audio/flac"}


def clips(directory: str = CLIPS) -> list[str]:
    out: list[str] = []
    for e in EXTS:
        out += glob.glob(os.path.join(directory, f"*.{e}"))
    return sorted(out)


def summarise(samples: list[float], meta: dict) -> dict:
    """Nearest-rank, the same percentile function D3's table uses -- two definitions of
    P95 in one report is one too many."""
    return {"stt_p50": round(pct(samples, 50), 1), "stt_p95": round(pct(samples, 95), 1),
            "stt_p100": round(pct(samples, 100), 1), "stt_n": len(samples), **meta}


def main():
    # both preconditions are reported together: finding out about the missing key only
    # after recording clips is a second trip for no reason.
    paths = clips()
    missing = []
    if not paths:
        missing.append(f"  * no clips in {CLIPS}/ -- drop a handful of real recordings there "
                       f"({', '.join(EXTS)}). Real speech, not a tone: round-trip time "
                       f"depends on what was said and how long it took to say it.")
    if not sarvam.API_KEY:
        missing.append("  * SARVAM_API_KEY is not set -- export it, or pass one on the "
                       "command line: `SARVAM_API_KEY=... make stt`.")
    if os.getenv("STT_PROVIDER") == "mock" or sarvam.provider() is sarvam.MockSTT:
        missing.append("  * STT_PROVIDER=mock is set -- a mock's 0.0 ms is not a measurement. "
                       "Unset it.")
    if missing:
        raise SystemExit("cannot measure the excluded legs:\n" + "\n".join(missing) +
                         "\n\nNothing is written until both hold. D3 keeps printing "
                         "'not measured on this run', which is the honest state of the "
                         "report until a real round trip has been timed.")

    samples, attempts = [], 0
    for p in paths:
        ext = p.rsplit(".", 1)[-1].lower()
        with open(p, "rb") as fh:
            audio = fh.read()
        t = sarvam.transcribe(audio, filename=os.path.basename(p),
                              content_type=MIME.get(ext, "application/octet-stream"))
        samples.append(t["stt_ms"])
        attempts += t["attempts"]
        print(f"  {os.path.basename(p):28.28s} {t['stt_ms']:7.1f} ms  [{t['language']}]  "
              f"{t['text'][:60]}", flush=True)

    out = summarise(samples, {"provider": "sarvam:batch", "model": sarvam.MODEL,
                              "clips": len(paths), "retries_used": attempts - len(paths),
                              "note": "measured outside the 200 ms budget, published with it"})
    os.makedirs(os.path.dirname(EXCLUDED), exist_ok=True)
    with open(EXCLUDED, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), f"-> {EXCLUDED}\n(re-run `make latency` to publish it)")


def demo():
    s = summarise([10.0, 20.0, 30.0, 40.0], {"provider": "x"})
    assert (s["stt_p50"], s["stt_p95"], s["stt_p100"], s["stt_n"]) == (20.0, 40.0, 40.0, 4), s
    assert "tts_p50" not in s, "this repo has no TTS stage -- it must not invent one"
    assert clips("/nonexistent") == []
    print("stt measure ok")


if __name__ == "__main__":
    demo() if "--selfcheck" in sys.argv else main()
