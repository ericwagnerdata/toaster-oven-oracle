"""
Load the FAISS index + chunks, and find the top-k chunks for a question.

Try it from the CLI:
    python src/retrieve.py "how do I reheat pizza?"
"""

from pathlib import Path
import pickle
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list[dict] | None = None


def _load():
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        _index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    if _chunks is None:
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            _chunks = pickle.load(f)


def retrieve(question: str, k: int = 5) -> list[dict]:
    """Return top-k chunks for the question, each with page, text, and score."""
    _load()
    q_vec = _model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, idxs = _index.search(q_vec, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        c = _chunks[i]
        results.append({"page": c["page"], "text": c["text"], "score": float(score)})
    return results


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "how do I reheat pizza?"
    print(f"Q: {question}\n")
    for i, r in enumerate(retrieve(question, k=5), start=1):
        print(f"--- #{i}  page {r['page']}  score={r['score']:.3f} ---")
        print(r["text"])
        print()
