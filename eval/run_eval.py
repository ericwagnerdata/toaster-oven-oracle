"""
Run the eval set end-to-end and print a pass/fail report.

Two metrics per question:
  retrieval  - did top-k chunks include any expected_pages? (recall@k)
  answer     - did the generated answer contain all must_contain substrings?

Usage:
    python eval/run_eval.py
"""

from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import answer  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.yaml"
TOP_K = 5


def main() -> None:
    with open(QUESTIONS_PATH) as f:
        cases = yaml.safe_load(f)

    retrieval_hits = 0
    answer_hits = 0
    total = len(cases)

    print(f"Running {total} cases (top_k={TOP_K})\n")
    print(f"{'retrieval':<10} {'answer':<8}  question")
    print("-" * 80)

    for c in cases:
        result = answer(c["question"], k=TOP_K)
        retrieved_pages = {chunk["page"] for chunk in result["chunks"]}
        expected = set(c.get("expected_pages") or [])
        answer_text = result["answer"].lower()

        if not expected:
            retrieval_ok = True  # out-of-domain questions have no target page
        else:
            retrieval_ok = bool(expected & retrieved_pages)

        must = [s.lower() for s in c.get("must_contain", [])]
        answer_ok = all(s in answer_text for s in must)

        retrieval_hits += int(retrieval_ok)
        answer_hits += int(answer_ok)

        r_mark = "PASS" if retrieval_ok else "FAIL"
        a_mark = "PASS" if answer_ok else "FAIL"
        print(f"{r_mark:<10} {a_mark:<8}  {c['question']}")
        if not answer_ok:
            missing = [s for s in must if s not in answer_text]
            print(f"                     missing: {missing}")
            print(f"                     got: {result['answer'][:120]}...")

    print("-" * 80)
    print(f"Retrieval recall@{TOP_K}: {retrieval_hits}/{total} "
          f"({retrieval_hits / total:.0%})")
    print(f"Answer correctness:    {answer_hits}/{total} "
          f"({answer_hits / total:.0%})")


if __name__ == "__main__":
    main()
