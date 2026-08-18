---
title: Mic RAG — the 200 ms window
emoji: 🎙️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Speak a question in Hindi, Tamil, Bengali or English — answered from MS MARCO inside a measured 200 ms budget.
---

# Mic RAG

Speak a question. It is transcribed, retrieved against `ai4bharat/MSMARCO-XI`, reranked, and
answered from the passage it cites — with the grounding checked before the answer is shown.

The page draws a **200 millisecond ruler**. Everything this system owns has to fit between
its walls: guards, embedding, retrieval, reranking, answer selection, grounding check.
Speech-to-text and speech-back are a vendor's clock, so they are drawn outside the walls at
the same scale, which is why they run off the frame. That boundary is the point of the
project, not a caveat about it.

**Measured, on the reference bench:** P50 **45.2 ms**, 1500/1500 requests inside the window.
Correct abstention **84%** against **3.8%** false abstention. Eight chunking strategies
compared on human relevance labels; the winner is metadata-filtered at 0.892 recall@50.

A refusal is a result. Ask it something out of corpus, or something unsafe, and it names the
gate that fired and the numbers that fired it.

Source, reports and the full engineering log: https://github.com/jitheender-ops/rag-model
