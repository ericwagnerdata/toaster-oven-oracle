"""
One-time script: PDF -> chunks -> embeddings -> FAISS index on disk.

Run once (or whenever the PDF or chunking settings change):
    python src/ingest.py
"""

from pathlib import Path
import pickle

import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "BOV900_USCM_IB_Rv1.pdf"
INDEX_DIR = ROOT / "data" / "faiss"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
LAST_ENGLISH_PAGE = 32  # manual is bilingual; FR starts at p.33
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_pages(pdf_path: Path) -> list[dict]:
    """Read PDF, return one record per page with page number + text."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        if i > LAST_ENGLISH_PAGE:
            break
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split each page into overlapping chunks; keep page number as metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append({"page": p["page"], "text": piece})
    return chunks


def embed(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Normalize so inner-product search = cosine similarity
    faiss.normalize_L2(vectors)
    return vectors.astype("float32")


def main() -> None:
    print(f"Loading {PDF_PATH.name}...")
    pages = load_pdf_pages(PDF_PATH)
    print(f"  {len(pages)} pages with text")

    print("Chunking...")
    chunks = chunk_pages(pages)
    print(f"  {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding chunks...")
    vectors = embed(chunks, model)
    print(f"  shape={vectors.shape}")

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done. Wrote {INDEX_DIR}/index.faiss and chunks.pkl")


if __name__ == "__main__":
    main()
