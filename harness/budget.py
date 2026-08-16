"""D2 -- the 200ms window as a runtime object the code obeys.

The window, stated once:
  excluded: mic + VAD, STT round trip | MEASURED: t0 -> t1 | excluded: TTS + network
  t0 = server receives STT is_final (stamped server-side)
  t1 = last answer token flushed to the socket, after the grounding verdict.

Overrun must be impossible, not unlikely: every stage asks before it runs.
"""
from __future__ import annotations

from harness.spans import NS_PER_MS, now_ns

TOTAL_MS = 200.0

# The degradation ladder. Checked in order, first match wins.
# (remaining_ms_below, name, what it does)
LADDER = [
    (40.0, "extractive", "no LLM: return the top chunk's best sentence, flagged extractive"),
    (60.0, "no_rerank", "skip the cross-encoder, serve the RRF order (~4 nDCG, buys 15ms)"),
    (150.0, "short_output", "output cap 96 -> 48 tokens, context trimmed to 2 chunks"),
]


class Budget:
    def __init__(self, total_ms: float = TOTAL_MS, t0_ns: int | None = None):
        self.total_ms = total_ms
        self.t0 = t0_ns or now_ns()
        self.violated: str | None = None
        self.degradations: list[str] = []

    def elapsed(self) -> float:
        return (now_ns() - self.t0) / NS_PER_MS

    def remaining(self) -> float:
        return self.total_ms - self.elapsed()

    def rung(self) -> str | None:
        """Which degradation rung the remaining budget puts us on, if any."""
        for below, name, _ in LADDER:
            if self.remaining() < below:
                return name
        return None

    def check(self, stage: str, need_ms: float, degradable: bool = False) -> str:
        """RUN | DEGRADE | SKIP -- decided before entering the stage."""
        left = self.remaining()
        if left >= need_ms:
            return "RUN"
        if degradable and left >= need_ms / 2:
            self.degradations.append(stage)
            return "DEGRADE"
        self.degradations.append(stage)
        return "SKIP"

    def report(self) -> dict:
        return {"total_ms": self.total_ms, "elapsed_ms": round(self.elapsed(), 3),
                "degraded": sorted(set(self.degradations)), "violated": self.violated}


def demo():
    import time

    b = Budget(total_ms=50)
    assert b.check("rerank", 15) == "RUN"
    assert Budget(total_ms=200).rung() is None
    time.sleep(0.045)
    assert b.remaining() < 15
    assert b.check("generate", 30) == "SKIP"
    assert b.check("rerank", 10, degradable=True) in ("DEGRADE", "SKIP")
    # deterministic rung check: pretend 155ms of the 200 are already spent
    b2 = Budget(total_ms=200, t0_ns=now_ns() - 155 * NS_PER_MS)
    assert b2.rung() == "no_rerank", (b2.remaining(), b2.rung())
    b3 = Budget(total_ms=200, t0_ns=now_ns() - 175 * NS_PER_MS)
    assert b3.rung() == "extractive", (b3.remaining(), b3.rung())
    print("budget ok", b.report())


if __name__ == "__main__":
    demo()
