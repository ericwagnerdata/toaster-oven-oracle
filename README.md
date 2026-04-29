# Toaster Oven Oracle

A local RAG over the Breville BOV900 instruction manual. Ask cooking questions, get answers grounded in the manual with page citations.

![The Oracle of BOV900](web/assets/Toaster%20Oven%20Oracle.png)

## What this is

A learning project — the goal is to build a small RAG end-to-end from the naive baseline up, then iterate. The corpus is one ~32-page PDF, so it fits comfortably in a local FAISS index and lets me focus on the pipeline rather than infra.

## Architecture

```text
PDF ──▶ page-by-page text ──▶ recursive splitter ──▶ chunks (500 char, 75 overlap)
                                                         │
                                                         ▼
                                          MiniLM-L6-v2 embeddings (384-d)
                                                         │
                                                         ▼
                                                   FAISS (flat L2)
                                                         │
                  question ──▶ embed ──▶ top-k=5 ────────┘
                                            │
                                            ▼
                            grounded prompt + excerpts
                                            │
                                            ▼
                             Claude Haiku 4.5 ──▶ answer + (p.N) citations
```

Code map:

- [src/ingest.py](src/ingest.py) — PDF → chunks → embeddings → FAISS index on disk (one-time)
- [src/retrieve.py](src/retrieve.py) — embed query, top-k chunks from FAISS
- [src/generate.py](src/generate.py) — retrieve + prompt + Claude call
- [src/app.py](src/app.py) — Streamlit UI
- [eval/run_eval.py](eval/run_eval.py) — harness over [eval/questions.yaml](eval/questions.yaml)
- [web/](web/) — design prototype (static HTML, mock retrieval)

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env             # add ANTHROPIC_API_KEY
python src/ingest.py             # one-time: build the FAISS index
streamlit run src/app.py         # the real UI
```

Or try the design prototype (mock answers, no backend):

```bash
python -m http.server 8000 --directory web
# open http://localhost:8000
```

CLI shortcuts:

```bash
python src/retrieve.py "how do I reheat pizza?"
python src/generate.py "how do I reheat pizza?"
python eval/run_eval.py
```

## What I learned / what I'd do differently

- **Naive-first paid off.** Flat FAISS + MiniLM + a 5-chunk top-k is a serviceable baseline on a 32-page corpus. Reaching for HNSW or a reranker before measuring would have been theatre.
- **Chunk size matters more than chunking strategy at this scale.** Recursive splitter at 500/75 hit the sweet spot — bigger chunks blurred the citations, smaller ones split tables.
- **Page-number grounding is a citation cheat code.** Forcing the model to cite `(p.N)` in the system prompt makes hallucinations much easier to spot during eval.
- **Bilingual PDFs are a trap.** The manual is English then French; embedding both halves polluted retrieval until I capped at the last English page.
- **Next iterations**: hybrid search (BM25 + dense), a cross-encoder reranker on the top 20, and a small Ragas-style faithfulness eval beyond the yes/no questions in [eval/questions.yaml](eval/questions.yaml).
