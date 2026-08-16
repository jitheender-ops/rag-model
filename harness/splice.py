"""make submit: splice the three report tables into README.md between markers.

The README never holds a number that was typed by hand -- it holds the same table the
report generated, or nothing.
"""
from __future__ import annotations

import os
import re
import sys

README = "README.md"
SECTIONS = {
    "D1": ("reports/chunking.md", r"\| strategy.*?\n(?:\|.*\n)+"),
    "D3": ("reports/latency.md", r"\| stage.*?\n(?:\|.*\n)+"),
    # the budget verdict, quoted verbatim from the report rather than retyped here: it is
    # the one sentence a reader looks for, and it is exactly the one worth never hand-editing
    "D3V": ("reports/latency.md", r"> \*\*[\d.]+ ms budget.*?\n"),
    "D4": ("reports/guardrails.md", r"\| metric.*?\n(?:\|.*\n)+"),
}


def extract(path: str, pattern: str) -> str:
    if not os.path.exists(path):
        return f"_not generated yet -- run the make target for {path}_"
    m = re.search(pattern, open(path, encoding="utf-8").read())
    return m.group().strip() if m else f"_no table found in {path}_"


def splice(readme: str = README) -> list[str]:
    text = open(readme, encoding="utf-8").read()
    done = []
    for key, (path, pattern) in SECTIONS.items():
        block = extract(path, pattern)
        start, end = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
        if start not in text or end not in text:
            continue
        text = re.sub(re.escape(start) + r".*?" + re.escape(end),
                      f"{start}\n{block}\n{end}", text, flags=re.S)
        done.append(key)
    open(readme, "w", encoding="utf-8").write(text)
    return done


def demo():
    p = "/tmp/_readme.md"
    open(p, "w").write("x\n<!-- D1:START -->\nold\n<!-- D1:END -->\ny\n")
    out = splice(p)
    body = open(p).read()
    assert "D1" in out and "old" not in body and body.startswith("x")
    print("splice ok", out)


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        demo()
    else:
        print("spliced:", ", ".join(splice()) or "no markers found")
