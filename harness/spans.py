"""Span recorder. Written first: D2 and D3 are both just readers of this file.

One clock: time.perf_counter_ns(). Never time.time() -- NTP correction
mid-benchmark silently invents milliseconds.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from functools import wraps

NS_PER_MS = 1_000_000


def now_ns() -> int:
    return time.perf_counter_ns()


class Trace:
    """One request. Holds its own spans; totals come from t0/t1, never from summing."""

    def __init__(self, qid: str, meta: dict | None = None):
        self.qid = qid
        self.meta = dict(meta or {})
        self.spans: dict[str, float] = {}
        self.events: list[dict] = []
        self.t0 = now_ns()
        self.t1: int | None = None

    @contextmanager
    def span(self, name: str):
        s = now_ns()
        try:
            yield
        finally:
            self.spans[name] = self.spans.get(name, 0.0) + (now_ns() - s) / NS_PER_MS

    def event(self, kind: str, **kw):
        self.events.append({"kind": kind, "at_ms": (now_ns() - self.t0) / NS_PER_MS, **kw})

    def close(self):
        if self.t1 is None:
            self.t1 = now_ns()
        return self

    @property
    def total_ms(self) -> float:
        return ((self.t1 or now_ns()) - self.t0) / NS_PER_MS

    def check_instrumentation(self, tol_ms: float = 2.0) -> bool:
        """Dev invariant: spans must account for the wall of the request."""
        return abs(sum(self.spans.values()) - self.total_ms) < tol_ms

    def row(self) -> dict:
        return {"qid": self.qid, "total_ms": round(self.total_ms, 3),
                "spans": {k: round(v, 3) for k, v in self.spans.items()},
                "events": self.events, **self.meta}


class Recorder:
    """Append-only JSONL sink. Store samples, never a pre-computed average."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, trace: Trace):
        self._fh.write(json.dumps(trace.close().row(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def stage(name: str, budget_ms: float | None = None, degradable: bool = False):
    """Decorate a pipeline stage: times it, and asks the budget before entering.

    The decorated fn takes (ctx, ...) where ctx has .trace and .budget.
    """

    def deco(fn):
        @wraps(fn)
        def wrapper(ctx, *a, **kw):
            if budget_ms is not None and ctx.budget is not None:
                verdict = ctx.budget.check(name, budget_ms, degradable)
                ctx.trace.meta.setdefault("verdicts", {})[name] = verdict
                if verdict == "SKIP":
                    ctx.trace.event("stage_skipped", stage=name)
                    return None
                kw["degraded"] = verdict == "DEGRADE"
            elif budget_ms is not None:
                kw["degraded"] = False
            with ctx.trace.span(name):
                out = fn(ctx, *a, **kw)
            if budget_ms is not None and ctx.budget is not None:
                spent = ctx.trace.spans.get(name, 0.0)
                if spent > 2 * budget_ms:
                    ctx.trace.event("deadline_violation", stage=name,
                                    spent_ms=round(spent, 3), budget_ms=budget_ms)
                    ctx.budget.violated = name
            return out

        wrapper.budget_ms = budget_ms
        wrapper.degradable = degradable
        return wrapper

    return deco


def demo():
    t = Trace("q1")
    with t.span("a"):
        time.sleep(0.002)
    with t.span("b"):
        time.sleep(0.001)
    t.close()
    assert t.total_ms >= 3.0, t.total_ms
    assert t.check_instrumentation(), (t.spans, t.total_ms)
    assert t.row()["qid"] == "q1"
    print("spans ok", t.row())


if __name__ == "__main__":
    demo()
