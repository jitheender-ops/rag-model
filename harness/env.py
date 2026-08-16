"""The shared spine: every report header stamps machine, region, provider, date, commit."""
from __future__ import annotations

import os
import platform
import random
import subprocess
import time

SEED = 42
DOTENV = ".env"


def load_dotenv(path: str = DOTENV) -> list[str]:
    """KEY=value lines into os.environ, without overwriting what is already set.

    Ten lines instead of python-dotenv, and it exists so a secret has one obvious home:
    a gitignored file, not a shell profile and not a command line that lands in shell
    history. Values already in the environment win, so `SARVAM_API_KEY=... make stt` still
    overrides the file for a one-off.
    """
    loaded = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded.append(key)
    except OSError:
        pass
    return loaded


load_dotenv()          # at import: every entry point in this repo imports harness.env


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


def demo():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, ".env")
        with open(path, "w") as fh:
            fh.write('# a comment\nFAKE_KEY_A = "value-a"\n\nFAKE_KEY_B=b\nnot a pair\n')
        os.environ["FAKE_KEY_B"] = "already-set"
        loaded = load_dotenv(path)
        assert os.environ["FAKE_KEY_A"] == "value-a", "quotes and spaces must be stripped"
        assert os.environ["FAKE_KEY_B"] == "already-set", "the environment must win over the file"
        assert loaded == ["FAKE_KEY_A"], loaded
        for k in ("FAKE_KEY_A", "FAKE_KEY_B"):
            os.environ.pop(k, None)
    assert load_dotenv(os.path.join("/nonexistent", ".env")) == [], "a missing file is fine"
    print("env ok (dotenv, stamp)")


if __name__ == "__main__":
    demo()
    print(header("stamp check"))
