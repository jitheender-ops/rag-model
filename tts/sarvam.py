"""Text-to-speech. Sarvam bulbul, the mirror image of stt/sarvam.py.

    synthesize("...", "hi")  ->  (wav_bytes, tts_ms)

WHERE THIS SITS, AND WHY IT IS NOT IN THE BUDGET
The 200 ms window ends at t1 = the last answer token flushed. Speaking that answer happens
after t1, so TTS is an excluded leg exactly as STT is an excluded leg before t0 -- measured,
published beside the budget in D3's report, and never added into it. Measured on this
account: 0.87 s of speech costs ~478 ms, 3.6 s costs ~983 ms. That is three to a hundred
times the entire measured window, which is the whole reason the window is drawn where it is.

WHAT IT COSTS TO GET WRONG
A voice assistant that speaks an answer the verifier refused is worse than one that stays
silent, so speak() takes the trace, not just a string: an abstention is spoken as the
refusal it is, never as an answer, and never silently dropped either.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
import wave

from harness.env import load_dotenv
from harness.spans import NS_PER_MS, now_ns
from stt.sarvam import API_KEY, STTError, _ctx

load_dotenv()

URL = "https://api.sarvam.ai/text-to-speech"
MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v2")
SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "anushka")
TIMEOUT_S = float(os.getenv("TTS_TIMEOUT_S", "30"))
RETRIES = int(os.getenv("TTS_RETRIES", "1"))
MAX_CHARS = 1500                      # the API caps a request; the answer cap is far below it

# our corpus languages -> the API's BCP-47 codes. Assamese has no bulbul voice, so it falls
# back to Bengali, its nearest script and the honest approximation rather than silence.
LANG = {"en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "bn": "bn-IN", "as": "bn-IN",
        "mr": "mr-IN", "te": "te-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
        "pa": "pa-IN", "or": "od-IN"}


class TTSError(RuntimeError):
    pass


def code_for(lang: str) -> str:
    return LANG.get((lang or "en").split("-")[0].lower(), "en-IN")


def synthesize(text: str, lang: str = "en", speaker: str = SPEAKER,
               model: str = MODEL, retries: int = RETRIES) -> tuple[bytes, float]:
    """(wav_bytes, tts_ms). Outside the 200 ms window by definition."""
    if not API_KEY:
        raise TTSError("SARVAM_API_KEY is not set -- put it in .env, or use TTS_PROVIDER=mock")
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        raise TTSError("nothing to speak")
    body = json.dumps({"text": text, "target_language_code": code_for(lang),
                       "speaker": speaker, "model": model}).encode()
    t0 = now_ns()
    last = None
    for attempt in range(1, retries + 2):
        req = urllib.request.Request(URL, data=body, method="POST")
        req.add_header("api-subscription-key", API_KEY)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_ctx()) as r:
                payload = json.loads(r.read())
            audios = payload.get("audios") or []
            if not audios:
                raise TTSError(f"no audio in the response: {list(payload)}")
            return base64.b64decode(audios[0]), (now_ns() - t0) / NS_PER_MS
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code < 500 and e.code != 429:      # client error: retrying cannot fix it
                break
        except TTSError:
            raise
        except Exception as e:
            last = repr(e)
            if "CERTIFICATE_VERIFY_FAILED" in last:
                raise TTSError("TLS verification failed -- see stt.sarvam.ssl_context()") from e
        if attempt <= retries:
            time.sleep(0.25 * attempt)
    raise TTSError(f"sarvam tts failed after {attempt} attempt(s): {last}")


def duration_s(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / (w.getframerate() or 1)


def silence(seconds: float = 0.4, rate: int = 22050) -> bytes:
    """A valid WAV of nothing. The mock, and what `speak` returns when there is no key."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def line_for(trace_meta: dict) -> str:
    """What to actually say, given a finished request.

    An abstention is spoken as a refusal. Reading out an empty answer is silence the user
    cannot distinguish from a crash, and reading out the top passage anyway would undo
    gate 4 at the last possible moment -- the point where nobody would look for it.
    """
    if not trace_meta.get("abstain"):
        return trace_meta.get("answer") or ""
    gate = (trace_meta.get("gate") or "").split("_")[0]
    return {
        "gate1": "I can't help with that request.",
        "gate2": "I don't have anything on that in my documents.",
        "gate4": "I found something related, but it doesn't answer your question, "
                 "so I'd rather not guess.",
    }.get(gate, "I don't have a grounded answer for that.")


def speak(trace_meta: dict, lang: str = "en") -> tuple[bytes, float, str]:
    """(wav, tts_ms, provider) for a finished request. TTS_PROVIDER=mock stays offline."""
    text = line_for(trace_meta)
    if os.getenv("TTS_PROVIDER") == "mock":
        return silence(0.3), 0.0, "mock"
    wav, ms = synthesize(text, lang)
    return wav, ms, f"sarvam:{MODEL}"


def demo():
    assert code_for("hi") == "hi-IN" and code_for("en") == "en-IN"
    assert code_for("as") == "bn-IN", "Assamese has no voice; Bengali is the stated fallback"
    assert code_for(None) == "en-IN" and code_for("zz") == "en-IN"

    # an abstention must never be spoken as if it were an answer
    assert line_for({"abstain": False, "answer": "the bridge opened in 1901"}) \
        == "the bridge opened in 1901"
    for gate, expect in (("gate1_unsafe", "can't help"), ("gate2_score", "don't have anything"),
                         ("gate4_nli", "doesn't answer")):
        said = line_for({"abstain": True, "gate": gate, "answer": ""})
        assert expect in said, (gate, said)
        assert said, "an abstention must still say something out loud"

    w = silence(0.25)
    assert w[:4] == b"RIFF" and 0.2 < duration_s(w) < 0.3, len(w)
    os.environ["TTS_PROVIDER"] = "mock"
    wav, ms, prov = speak({"abstain": True, "gate": "gate2_score"}, "bn")
    assert prov == "mock" and ms == 0.0 and wav[:4] == b"RIFF"
    del os.environ["TTS_PROVIDER"]

    if not API_KEY:                    # missing key must be loud, never a silent mock
        try:
            synthesize("x")
            raise AssertionError("missing key must raise")
        except TTSError as e:
            assert "SARVAM_API_KEY" in str(e)
    print("tts ok (lang map, refusal lines, mock, key guard)")


if __name__ == "__main__":
    demo()
