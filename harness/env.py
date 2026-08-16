"""The shared spine: every report header stamps machine, region, provider, date, commit."""
from __future__ import annotations

import os
import platform
import random
import subprocess
import time

SEED = 42


def pin_randomness(extra: int = 0):
    """Seed 42 across sampling and shuffles; LLM temperature 0 is set at the call site."""
    random.seed(SEED + extra)


def _sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def embedder() -> str:
    """Which embedder produced these numbers -- the single biggest lever on every table."""
    from d1.index import BACKEND, DIM, MODEL_NAME
    return (f"{MODEL_NAME} / {DIM()}d" if BACKEND == "st" else "hashed-bow / 4096d")


def stamp() -> dict:
    return {
        "machine": os.getenv("BENCH_MACHINE", f"{platform.machine()} / {os.cpu_count()} vCPU"),
        "region": os.getenv("BENCH_REGION", "local"),
        "provider": os.getenv("BENCH_PROVIDER", platform.system()),
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "commit": _sh("git", "rev-parse", "--short", "HEAD") or "uncommitted",
        "embedder": embedder(),
        "python": platform.python_version(),
        "seed": SEED,
    }


def header(title: str, extra: dict | None = None) -> str:
    s = {**stamp(), **(extra or {})}
    rows = "\n".join(f"| {k} | {v} |" for k, v in s.items())
    return f"# {title}\n\n| field | value |\n|---|---|\n{rows}\n"


if __name__ == "__main__":
    print(header("stamp check"))
