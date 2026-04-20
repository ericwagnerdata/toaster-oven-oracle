"""
Ask a question: retrieve chunks, build a grounded prompt, call Claude, return the answer.

CLI:
    python src/generate.py "how do I reheat pizza?"
"""

from pathlib import Path
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = "claude-haiku-4-5-20251001"
TOP_K = 5

SYSTEM_PROMPT = """You are an assistant that answers questions about a Breville BOV900 \
toaster oven, using ONLY the provided excerpts from its instruction manual.

Rules:
- Answer concisely. When giving settings, include the exact values (temperature, time, rack position, function).
- Cite the page number for every factual claim, like (p.23).
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Prefer English excerpts over French ones when both are present.
"""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[page {c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"""Manual excerpts:

{context}

---

Question: {question}

Answer using only the excerpts above. Cite pages like (p.N)."""


def answer(question: str, k: int = TOP_K) -> dict:
    chunks = retrieve(question, k=k)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, chunks)}],
    )
    return {
        "question": question,
        "answer": resp.content[0].text,
        "chunks": chunks,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "how do I reheat pizza?"
    result = answer(q)
    print(f"Q: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for c in result["chunks"]:
        print(f"  p.{c['page']}  score={c['score']:.3f}")
