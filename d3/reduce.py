"""D3 stage 4: reduce the trace log to the tables and the chart.

    make latency   ->  reports/latency.md + reports/latency.svg

Three rules this file exists to enforce:
  * nearest-rank percentiles on the sorted samples, no interpolation;
  * per-stage P50s do not sum to the end-to-end P50 -- the total comes from each
    request's own total, never by adding columns;
  * P100 over 500 samples is one observation. It is reported because it was asked for,
    with P95 and P99 beside it and the stage that caused it named.

One script emits the markdown and the chart, so they can never disagree.
"""
from __future__ import annotations

import json
import math
import os

from harness.budget import TOTAL_MS
from harness.env import header

TRACES = "artifacts/d3"
STAGES = ["input_guards", "embed_query", "retrieve", "rerank", "generate", "verify"]
LABEL = {"input_guards": "input guards", "embed_query": "embed query",
         "retrieve": "dense + bm25 + rrf", "rerank": "rerank",
         "generate": "generate", "verify": "verify"}
EXCLUDED = "data/excluded_legs.json"


def pct(samples: list[float], p: float) -> float:
    """Nearest-rank: the smallest value at or above p% of the sorted samples."""
    if not samples:
        return float("nan")
    s = sorted(samples)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def load(mode: str) -> list[dict]:
    path = f"{TRACES}/traces_{mode}.jsonl"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def table(rows: list[dict], cold: list[dict]) -> str:
    out = ["| stage             |  P50 |  P70 |  P95 | P100 |",
           "|-------------------|------|------|------|------|"]
    for st in STAGES:
        s = [r["spans"][st] for r in rows if st in r["spans"]]
        if not s:
            continue
        out.append(f"| {LABEL[st]:17s} | {pct(s, 50):4.2f} | {pct(s, 70):4.2f} | "
                   f"{pct(s, 95):4.2f} | {pct(s, 100):4.2f} |")
    for name, rs in (("END-TO-END (warm)", rows), ("END-TO-END (cold)", cold)):
        if not rs:
            continue
        tot = [r["total_ms"] for r in rs]        # each request's own total, never a column sum
        out.append(f"| {name:17s} | {pct(tot, 50):4.2f} | {pct(tot, 70):4.2f} | "
                   f"{pct(tot, 95):4.2f} | {pct(tot, 100):4.2f} |")
    return "\n".join(out)


def attribute_max(rows: list[dict]) -> str:
    if not rows:
        return "no samples"
    worst = max(rows, key=lambda r: r["total_ms"])
    stage = max(worst["spans"].items(), key=lambda kv: kv[1]) if worst["spans"] else ("--", 0)
    fired = [e["kind"] for e in worst.get("events", [])] or ["none"]
    return (f"P100 = {worst['total_ms']:.1f} ms on `{worst['qid']}` ({worst.get('kind', '?')}, "
            f"{worst.get('lang', '?')}); the dominant stage was **{stage[0]}** at "
            f"{stage[1]:.1f} ms, fallbacks fired: {', '.join(fired)}")


def svg(rows: list[dict], cold: list[dict], path: str):
    """Per-stage P50/P95 bars + the 200ms line. Stdlib SVG, not a PNG -- same script, same
    numbers as the table above it."""
    stages = [st for st in STAGES if any(st in r["spans"] for r in rows)]
    p50 = [pct([r["spans"][st] for r in rows if st in r["spans"]], 50) for st in stages]
    p95 = [pct([r["spans"][st] for r in rows if st in r["spans"]], 95) for st in stages]
    e2e = pct([r["total_ms"] for r in rows], 50) if rows else 0
    w, h, pad = 720, 320, 46
    scale = (w - 2 * pad) / max(TOTAL_MS, max(p95 + [e2e], default=1))
    bars = []
    for i, st in enumerate(stages):
        y = pad + i * 34
        bars.append(f'<rect x="{pad}" y="{y}" width="{p95[i] * scale:.1f}" height="12" '
                    f'fill="#cbd5e1"/>'
                    f'<rect x="{pad}" y="{y}" width="{p50[i] * scale:.1f}" height="12" '
                    f'fill="#2563eb"/>'
                    f'<text x="4" y="{y + 11}" font-size="11" fill="#334155">{LABEL[st]}</text>'
                    f'<text x="{pad + max(p95[i] * scale, 2) + 6}" y="{y + 11}" font-size="10" '
                    f'fill="#64748b">{p50[i]:.1f} / {p95[i]:.1f} ms</text>')
    budget_x = pad + TOTAL_MS * scale
    svg_doc = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="white"/>'
        f'<text x="{pad}" y="24" font-size="14" fill="#0f172a">Per-stage latency, warm, '
        f'n={len(rows)} (bar = P50, shadow = P95)</text>'
        + "".join(bars) +
        f'<line x1="{budget_x}" y1="{pad - 8}" x2="{budget_x}" y2="{h - 40}" '
        f'stroke="#dc2626" stroke-dasharray="4 3"/>'
        f'<text x="{budget_x + 4}" y="{h - 44}" font-size="10" fill="#dc2626">'
        f'{TOTAL_MS:.0f} ms budget</text>'
        f'<text x="{pad}" y="{h - 16}" font-size="11" fill="#0f172a">end-to-end P50 '
        f'{e2e:.1f} ms &#183; cold P50 '
        f'{pct([r["total_ms"] for r in cold], 50) if cold else float("nan"):.1f} ms</text>'
        f'</svg>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg_doc)


def backend_note() -> str:
    """Say which components are real, so nobody reads a stand-in's timing as a model's."""
    from d1.index import BACKEND
    real = ("The `embed query` row is a real transformer forward pass "
            "(multilingual-e5-small on CPU). Generation is still extractive and the "
            "reranker is lexical, so those two rows are floors, not an LLM's cost: "
            "budget for ~15 ms of cross-encoder and the generator's own time on top."
            if BACKEND == "st" else
            "This run used the hashed bag-of-words embedder, so every row is the "
            "harness's own cost rather than a model's. Run with the venv "
            "(`make venv`) for the transformer numbers.")
    return "\n_All stages are milliseconds. " + real + "_\n"


def cache_caveat(warm: list[dict], cache: float) -> str:
    """Only claim the cache is flattering the P50 when the numbers actually say so."""
    if cache <= 0.05:
        return ""
    e2e = pct([r["total_ms"] for r in warm], 50)
    worst_stage = max((pct([r["spans"][st] for r in warm if st in r["spans"]], 50)
                       for st in STAGES if any(st in r["spans"] for r in warm)), default=0.0)
    if e2e < worst_stage:
        return (f"\n\n> The warm end-to-end P50 ({e2e:.2f} ms) sits **below the slowest "
                f"stage's own P50** ({worst_stage:.2f} ms) because {cache:.1%} of warm "
                f"queries are served from the semantic cache and never enter a stage at "
                f"all. That is the cache flattering the headline number, which is why the "
                f"cache-off cold row is published beside it -- read the cold row as the "
                f"real cost of a first-time question.")
    return (f"\n\n> {cache:.1%} of warm queries hit the semantic cache. The cache-off "
            f"cold row is published beside the warm one; read it as the cost of a "
            f"first-time question.")


def excluded_legs() -> str:
    if os.path.exists(EXCLUDED):
        e = json.load(open(EXCLUDED))
        src = (f" over {e['stt_n']} clips via {e.get('provider', '?')}"
               if e.get("stt_n") else "")
        return (f"excluded legs (P50 / P95 / P100, ms): STT {e.get('stt_p50', '_')} / "
                f"{e.get('stt_p95', '_')} / {e.get('stt_p100', '_')}{src}    "
                f"TTS {e.get('tts_p50', '_')}    client RTT {e.get('client_rtt', '_')}    "
                f"(`_` = no such stage in this repo, or not measured)")
    return ("excluded legs: not measured on this run -- `make stt` with SARVAM_API_KEY set "
            f"and clips in data/audio/ writes `{EXCLUDED}`, and this line becomes the "
            "measurement. They are excluded from the budget, not hidden from the report.")


def verdict(warm: list[dict], cold: list[dict], conc: list[dict]) -> str:
    """The one line the budget exists to produce, at the top where it cannot be missed.

    It counts every recorded request in every mode -- warm, cold and concurrent. Reporting
    PASS on the warm run alone would be picking the friendliest of three, and the concurrent
    pass is the one that queues.
    """
    rows = [(name, rs) for name, rs in (("warm", warm), ("cold", cold), ("conc", conc)) if rs]
    over = {name: sum(1 for r in rs if r["total_ms"] > TOTAL_MS) for name, rs in rows}
    n = sum(len(rs) for _, rs in rows)
    total_over = sum(over.values())
    worst = max((pct([r["total_ms"] for r in rs], 100) for _, rs in rows), default=0.0)
    detail = ", ".join(f"{name} {over[name]}/{len(rs)}" for name, rs in rows)
    if total_over == 0:
        return (f"\n> **{TOTAL_MS:.0f} ms budget: PASS.** {n}/{n} requests inside the window "
                f"across every mode ({detail} over budget). Slowest single request "
                f"{worst:.1f} ms. Excluded legs are listed below and are not part of this "
                f"verdict.\n")
    return (f"\n> **{TOTAL_MS:.0f} ms budget: {total_over} of {n} requests over.** "
            f"By mode: {detail}. Slowest single request {worst:.1f} ms. The degradation "
            f"ladder fires before a stage runs, so an overrun here is a bug worth naming, "
            f"not a tuning knob.\n")


def main():
    warm, cold, conc = load("warm"), load("cold"), load("conc")
    if not warm:
        raise SystemExit("no warm traces -- run `python3 d3/run.py` first")
    deg = sum(1 for r in warm if r.get("degraded")) / len(warm)
    cache = sum(1 for r in warm if r.get("cache_hit")) / len(warm)
    over = [r for r in warm if r["total_ms"] > TOTAL_MS]
    tot = [r["total_ms"] for r in warm]
    md = [
        header("D3 - Latency analytics", {"n": len(warm), "budget": f"{TOTAL_MS:.0f} ms"}),
        verdict(warm, cold, conc),
        "\n## The window\n",
        "```\n"
        "  [ mic + VAD ]   [ STT round trip ]   |=== t0 ---> t1 MEASURED ===|   [ TTS + net ]\n"
        "     excluded          excluded          guards -> retrieve -> rerank      excluded\n"
        "                                         -> generate -> verify\n"
        "  t0 = server receives the STT is_final event, stamped server-side\n"
        "  t1 = last answer token flushed to the socket, after the grounding verdict\n"
        "```\n",
        excluded_legs(),
        "\n\n## Table shape - warm, n = %d\n" % len(warm),
        table(warm, cold),
        f"\ndegradation rate: {deg:.1%}   cache hit rate: {cache:.1%}   "
        f"n={len(warm)}, seed 42   over-budget: {len(over)}/{len(warm)} "
        f"({len(over) / len(warm):.1%})",
        cache_caveat(warm, cache),
        "\n\n## P100, said out loud\n",
        f"P100 over {len(warm)} samples is one observation: it is the max and it is unstable "
        f"by construction. P95 = {pct(tot, 95):.1f} ms, P99 = {pct(tot, 99):.1f} ms, "
        f"P100 = {pct(tot, 100):.1f} ms. " + attribute_max(warm) + ".",
        "\n\n## Concurrency\n",
        (f"A separate `--concurrency 4` pass over the same 500 queries: P50 "
         f"{pct([r['total_ms'] for r in conc], 50):.1f} ms, P95 "
         f"{pct([r['total_ms'] for r in conc], 95):.1f} ms, over-budget "
         f"{sum(1 for r in conc if r['total_ms'] > TOTAL_MS)}/{len(conc)}."
         if conc else "not run -- `python3 d3/run.py --only conc`."),
        "\n\n![per-stage latency](latency.svg)\n",
        backend_note(),
        "\n_Cold = fresh process, semantic cache off, encoder already resident: a server "
        "loads its model before it accepts traffic, so that ~10 s belongs to startup and "
        "not to the first caller's 200 ms. Warm = 50 discarded warmups first, "
        "cache on; the cache-off run is the `cold` row and the cache hit rate is printed "
        "above, so a repeat-heavy query file cannot flatter the P50 unnoticed._\n",
    ]
    os.makedirs("reports", exist_ok=True)
    with open("reports/latency.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    svg(warm, cold, "reports/latency.svg")
    print(f"warm P50 {pct(tot, 50):.1f} ms  P95 {pct(tot, 95):.1f}  P100 {pct(tot, 100):.1f}  "
          f"-> reports/latency.md + latency.svg")


def demo():
    s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert pct(s, 50) == 5 and pct(s, 100) == 10 and pct(s, 70) == 7
    assert pct([4], 95) == 4 and math.isnan(pct([], 50))
    # the verdict must count every mode, not just the friendliest one
    fast, slow = [{"total_ms": 10.0}], [{"total_ms": TOTAL_MS + 1}]
    assert "PASS" in verdict(fast, fast, fast)
    v = verdict(fast, slow, fast)
    assert "PASS" not in v and "1 of 3 requests over" in v, v
    # per-stage P50s must not be summed into an end-to-end P50
    rows = [{"spans": {"a": 10, "b": 1}, "total_ms": 11},
            {"spans": {"a": 1, "b": 10}, "total_ms": 11}]
    assert pct([r["spans"]["a"] for r in rows], 50) + pct([r["spans"]["b"] for r in rows], 50) \
        != pct([r["total_ms"] for r in rows], 50)
    print("percentiles ok")


if __name__ == "__main__":
    import sys

    demo() if "--selfcheck" in sys.argv else main()
