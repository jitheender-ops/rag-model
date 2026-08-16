"""Requirement 1: speech-to-text. Sarvam, picked for Indic + code-switched speech.

Two paths, one interface:

  transcribe(audio_bytes, ...)          batch  POST https://api.sarvam.ai/speech-to-text
  StreamingSession                      live   wss://api.sarvam.ai/speech-to-text/ws

WHERE t0 COMES FROM, AND WHY STT IS NOT IN THE BUDGET
The 200 ms budget starts at t0 = the instant the server holds a final transcript, stamped
server-side (harness/spans.Trace is created at that moment, never before). The STT leg is
measured and published beside the budget as an excluded leg -- we are not hiding a
vendor's network latency, we are refusing to let it decide whether our retrieval
engineering passes. Sarvam's WebSocket has no `is_final` field: the transcript refines
progressively and `{"type":"flush"}` finalises it, so the flush response is our is_final
and that is stated rather than implied.

RETRIES LIVE HERE, NOT IN THE PIPELINE
A retry inside a 200 ms deadline is arithmetic that does not work: one retry of a 30 ms
stage plus its timeout is the whole budget. So the harness retries where retrying is
affordable -- the STT call, which is outside the window -- and inside the window it
degrades instead (harness/budget.py's ladder). That split is the design, not an omission.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid

from harness.env import load_dotenv
from harness.spans import NS_PER_MS, now_ns

load_dotenv()          # a key in .env counts as set, so nothing needs exporting

API_KEY = os.getenv("SARVAM_API_KEY", "")
BATCH_URL = "https://api.sarvam.ai/speech-to-text"
WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"
MODEL = os.getenv("SARVAM_MODEL", "saaras:v3")
LANGUAGE = os.getenv("SARVAM_LANGUAGE", "unknown")      # auto-detect across hi/ta/bn/en
RETRIES = int(os.getenv("STT_RETRIES", "2"))
TIMEOUT_S = float(os.getenv("STT_TIMEOUT_S", "20"))


class STTError(RuntimeError):
    pass


def ssl_context():
    """A context that can actually verify api.sarvam.ai.

    macOS python.org and uv-built interpreters ship no CA bundle -- ssl's default cafile is
    None -- so urllib fails every HTTPS request with CERTIFICATE_VERIFY_FAILED and it reads
    like the vendor is down. certifi is already here (sentence-transformers depends on it);
    when it is not, fall back to the system default rather than to no verification, because
    an STT client that silently stops checking certificates is a worse bug than a broken one.
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = None


def _ctx():
    global _SSL
    if _SSL is None:
        _SSL = ssl_context()
    return _SSL


# magic bytes -> mime. The extension is a claim; the first bytes are the fact, and the two
# disagree constantly: phone recorders and voice-memo exports happily write MP3 into a .wav
# name, and the vendor answers a mislabelled upload with "please check the audio format",
# which reads like the recording is broken rather than misnamed.
MAGIC = (
    (b"RIFF", "audio/wav"), (b"OggS", "audio/ogg"), (b"fLaC", "audio/flac"),
    (b"ID3", "audio/mpeg"), (b"\x1a\x45\xdf\xa3", "audio/webm"),
)


def content_type_of(audio: bytes, filename: str = "") -> str:
    """What this actually is, falling back to the extension only when the bytes are mute."""
    head = audio[:16]
    for sig, mime in MAGIC:
        if head.startswith(sig):
            return mime
    if len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "audio/mpeg"                     # bare MPEG frame sync, no ID3 tag
    if head[4:8] == b"ftyp":
        return "audio/mp4"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4", "webm": "audio/webm",
            "ogg": "audio/ogg", "flac": "audio/flac"}.get(ext, "application/octet-stream")


def ext_for(mime: str) -> str:
    return {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/mp4": "m4a",
            "audio/webm": "webm", "audio/ogg": "ogg", "audio/flac": "flac"}.get(mime, "bin")


class Transcript(dict):
    """{text, language, stt_ms, attempts, provider}. A dict so it serialises straight
    into the trace log."""


def _multipart(fields: dict[str, str], filename: str, audio: bytes,
               content_type: str) -> tuple[bytes, str]:
    boundary = f"----mic-rag-{uuid.uuid4().hex}"
    out = []
    for k, v in fields.items():
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                   f"{v}\r\n".encode())
    out.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode())
    out.append(audio)
    out.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def transcribe(audio: bytes, filename: str = "audio.webm",
               content_type: str = "audio/webm", language: str = LANGUAGE,
               model: str = MODEL, retries: int = RETRIES) -> Transcript:
    """Batch transcription with bounded retries. Outside the 200 ms window by definition."""
    if not API_KEY:
        raise STTError("SARVAM_API_KEY is not set -- export it, or run with STT_PROVIDER=mock")
    fields = {"model": model, "language_code": language}
    body, ctype = _multipart(fields, filename, audio, content_type)
    t0 = now_ns()
    last = None
    for attempt in range(1, retries + 2):
        req = urllib.request.Request(BATCH_URL, data=body, method="POST")
        req.add_header("api-subscription-key", API_KEY)
        req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_ctx()) as r:
                payload = json.loads(r.read())
            return Transcript(text=payload.get("transcript", "").strip(),
                              language=payload.get("language_code", language),
                              confidence=payload.get("language_probability"),
                              stt_ms=(now_ns() - t0) / NS_PER_MS,
                              attempts=attempt, provider="sarvam:batch")
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
            if e.code < 500 and e.code != 429:      # client error: retrying cannot fix it
                break
        except Exception as e:                       # timeout, DNS, reset -- worth a retry
            last = repr(e)
            if "CERTIFICATE_VERIFY_FAILED" in last:
                # not transient: three identical failures and a 0.75 s backoff teach nobody
                # anything, and the message that matters is the one about the CA bundle
                raise STTError(
                    "TLS verification failed against the Sarvam API. This interpreter has no "
                    "CA bundle (ssl's default cafile is None, which is normal for macOS "
                    "python.org and uv builds). `uv pip install --python .venv/bin/python "
                    "certifi` fixes it; sarvam.ssl_context() picks it up automatically."
                ) from e
        if attempt <= retries:
            time.sleep(0.25 * attempt)               # linear backoff, bounded
    raise STTError(f"sarvam batch failed after {attempt} attempt(s): {last}")


class StreamingSession:
    """Live mic path. Feed PCM/WAV chunks, then finalise().

    finalise() returns the moment Sarvam answers the flush -- that return is what the
    server treats as `is_final`, and the caller stamps t0 immediately after it.
    """

    def __init__(self, language: str = LANGUAGE, model: str = MODEL,
                 sample_rate: int = 16000, codec: str = "pcm_s16le"):
        self.url = (f"{WS_URL}?language-code={language}&model={model}"
                    f"&sample_rate={sample_rate}&input_audio_codec={codec}&vad_signals=true")
        self.sample_rate = sample_rate
        self.ws = None
        self.t_first_chunk = None
        self.partials: list[str] = []

    async def __aenter__(self):
        import websockets
        if not API_KEY:
            raise STTError("SARVAM_API_KEY is not set")
        self.ws = await websockets.connect(
            self.url, additional_headers={"Api-Subscription-Key": API_KEY})
        return self

    async def __aexit__(self, *exc):
        if self.ws:
            await self.ws.close()

    async def send_audio(self, pcm: bytes, encoding: str = "audio/wav"):
        if self.t_first_chunk is None:
            self.t_first_chunk = now_ns()
        await self.ws.send(json.dumps({"audio": {
            "data": base64.b64encode(pcm).decode(),
            "sample_rate": str(self.sample_rate),
            "encoding": encoding}}))

    async def finalise(self, timeout_s: float = TIMEOUT_S) -> Transcript:
        """Send flush, drain until the transcript stops refining. The response to flush is
        our is_final -- Sarvam does not send one, so we define it and say so."""
        import asyncio
        await self.ws.send(json.dumps({"type": "flush"}))
        text, lang = "", LANGUAGE
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(self.ws.recv(),
                                             timeout=max(0.05, deadline - time.monotonic()))
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            if msg.get("type") == "error":
                raise STTError(str(msg.get("data")))
            if msg.get("type") == "data":
                d = msg.get("data", {})
                if d.get("transcript"):
                    text = d["transcript"].strip()
                    lang = d.get("language_code", lang)
                    self.partials.append(text)
                    break                    # flush answered: this is the final transcript
        started = self.t_first_chunk or now_ns()
        return Transcript(text=text, language=lang,
                          stt_ms=(now_ns() - started) / NS_PER_MS,
                          attempts=1, provider="sarvam:ws")


class MockSTT:
    """No key, no network: used by the self-checks and by `make demo` so the whole voice
    path is exercisable offline. It is never silently substituted for the real one --
    STT_PROVIDER=mock has to be asked for."""

    @staticmethod
    def transcribe(audio: bytes, **kw) -> Transcript:
        text = kw.get("text") or os.getenv("MOCK_TRANSCRIPT", "what is a corporation")
        return Transcript(text=text, language="en-IN", stt_ms=0.0, attempts=1,
                          provider="mock")


def provider():
    """sarvam unless STT_PROVIDER=mock. Chosen once, reported in every trace."""
    return MockSTT if os.getenv("STT_PROVIDER") == "mock" else _Sarvam


class _Sarvam:
    transcribe = staticmethod(transcribe)


def demo():
    body, ctype = _multipart({"model": "saaras:v3"}, "a.wav", b"RIFFdata", "audio/wav")
    assert b'name="model"' in body and b"RIFFdata" in body
    assert ctype.startswith("multipart/form-data; boundary=")
    assert body.rstrip().endswith(b"--")

    # the bytes decide, not the name: an MP3 called .wav must be sent as audio/mpeg
    assert content_type_of(b"RIFF\x00\x00\x00\x00WAVE", "x.mp3") == "audio/wav"
    assert content_type_of(b"ID3\x03\x00lots of tag", "x.wav") == "audio/mpeg"
    assert content_type_of(b"\xff\xfb\x90\x00frame", "x.wav") == "audio/mpeg"
    assert content_type_of(b"\x1a\x45\xdf\xa3seg", "x.bin") == "audio/webm"
    assert content_type_of(b"\x00\x00\x00 ftypM4A ", "x.bin") == "audio/mp4"
    assert content_type_of(b"", "x.flac") == "audio/flac", "mute bytes fall back to the name"
    assert content_type_of(b"", "x") == "application/octet-stream"

    t = MockSTT.transcribe(b"", text="कॉर्पोरेशन क्या है")
    assert t["text"] == "कॉर्पोरेशन क्या है" and t["provider"] == "mock"

    os.environ["STT_PROVIDER"] = "mock"
    assert provider() is MockSTT
    del os.environ["STT_PROVIDER"]
    assert provider() is _Sarvam

    if not API_KEY:                       # the failure has to be loud, not a silent mock
        try:
            transcribe(b"x")
            raise AssertionError("missing key must raise")
        except STTError as e:
            assert "SARVAM_API_KEY" in str(e)
    print("stt ok (batch multipart, mock, key guard)")


if __name__ == "__main__":
    demo()
