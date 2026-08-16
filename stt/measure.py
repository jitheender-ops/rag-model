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
  * no clips, no STT numbers -- the TTS half is still written, and the report says which
    half is missing rather than quietly showing one leg as if it were both.
  * only what was actually measured is written. Whichever leg cannot be measured is absent
    rather than invented, and D3 prints "_" for it.

BOTH LEGS, ONE FILE
STT sits before t0 and TTS after t1, so neither is inside the 200 ms window and both belong
beside it. TTS needs no clips -- only text -- so a key with no recordings still measures
half of this file, and the report says which half.
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


def answer_shaped_text(n: int = 8) -> list[tuple[str, str]]:
    """(lang, text) pairs the length of a real answer, taken from the frozen corpus.

    Not lorem ipsum and not the questions: TTS cost scales with how much speech it has to
    produce, and this system speaks answers. One per language, first sentence of a passage,
    deterministic."""
    from d1.chunkers import sentences
    from d1 import corpus
    out, seen = [], set()
    for d in corpus.load():
        if d["lang"] in seen or not d["passages"]:
            continue
        first = sentences(d["passages"][0]["text"])
        if not first:
            continue
        seen.add(d["lang"])
        out.append((d["lang"], first[0][0][:300]))
        if len(out) >= n:
            break
    return out


def measure_tts() -> dict:
    """The leg after t1. Empty when there is no key, which the caller reports as unmeasured."""
    from tts import sarvam as tts
    samples = []
    for lang, text in answer_shaped_text():
        _wav, ms = tts.synthesize(text, lang)
        samples.append(ms)
        print(f"  tts {lang:3s} {ms:7.1f} ms  {len(text):>3} chars  {text[:44]}", flush=True)
    if not samples:
        return {}
    return {"tts_p50": round(pct(samples, 50), 1), "tts_p95": round(pct(samples, 95), 1),
            "tts_p100": round(pct(samples, 100), 1), "tts_n": len(samples),
            "tts_provider": f"sarvam:{tts.MODEL}"}


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
    # a key with no clips still measures the TTS leg; only a missing key blocks everything
    if not sarvam.API_KEY or os.getenv("STT_PROVIDER") == "mock":
        raise SystemExit("cannot measure either excluded leg:\n" + "\n".join(missing) +
                         "\n\nNothing is written without a key. D3 keeps printing 'not "
                         "measured on this run', which is the honest state of the report "
                         "until a real round trip has been timed.")
    if not paths:
        print("no clips in %s/ -- measuring the TTS leg only; STT stays unmeasured.\n"
              "  (data/audio/RECORD_THESE.md lists ten worth recording)\n" % CLIPS)

    samples, attempts = [], 0
    for p in paths:
        with open(p, "rb") as fh:
            audio = fh.read()
        mime = sarvam.content_type_of(audio, p)
        # send it under the name its bytes deserve: a .wav holding MP3 is common, and the
        # vendor rejects the mismatch with a message about the audio rather than the label
        t = sarvam.transcribe(audio, filename=f"{os.path.basename(p)}.{sarvam.ext_for(mime)}",
                              content_type=mime)
        samples.append(t["stt_ms"])
        attempts += t["attempts"]
        print(f"  {os.path.basename(p):28.28s} {t['stt_ms']:7.1f} ms  [{t['language']}]  "
              f"{t['text'][:60]}", flush=True)

    out = ({} if not samples else
           summarise(samples, {"provider": "sarvam:batch", "model": sarvam.MODEL,
                               "clips": len(paths), "retries_used": attempts - len(paths)}))
    out.update(measure_tts())
    out["note"] = ("both legs measured outside the 200 ms budget and published beside it: "
                   "STT before t0, TTS after t1")
    os.makedirs(os.path.dirname(EXCLUDED), exist_ok=True)
    with open(EXCLUDED, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), f"-> {EXCLUDED}\n(re-run `make latency` to publish it)")


def demo():
    s = summarise([10.0, 20.0, 30.0, 40.0], {"provider": "x"})
    assert (s["stt_p50"], s["stt_p95"], s["stt_p100"], s["stt_n"]) == (20.0, 40.0, 40.0, 4), s
    assert "tts_p50" not in s, "summarise() reports the STT leg only; TTS is measured apart"
    assert clips("/nonexistent") == []

    # answer-shaped, one per language, and actually from the corpus
    texts = answer_shaped_text(4)
    assert texts and len({lang for lang, _ in texts}) == len(texts), texts
    assert all(t.strip() for _, t in texts), texts
    print("stt measure ok")


if __name__ == "__main__":
    demo() if "--selfcheck" in sys.argv else main()
