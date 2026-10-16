"""Retrieval quality check: does the right page show up in the top chunks?

Each entry in questions.json names a question and the pages that answer it.
A question is a "hit" when at least one retrieved chunk comes from one of those
pages. Entries with "source": null have no answer in the documents; for them we
only record how similar the closest chunk is, to see whether a minimum-similarity
cutoff could tell them apart from real questions. No chat model is called.

Usage (from the repository root):
    python -m evals.eval_retrieval [questions file]
Results are printed and saved to evals/results/ so runs can be compared.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import chromadb

from ask import retrieve
from config import BACKEND, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DB_DIR, TOP_K
from llm_client import LLMError

HERE = Path(__file__).parent


def first_hit_rank(hits, case):
    """1-based rank of the first chunk from an expected page, or None."""
    for rank, (_, meta, _) in enumerate(hits, start=1):
        if meta["source"] == case["source"] and meta["page"] in case["pages"]:
            return rank
    return None


def evaluate(case, hits):
    """One result row: what was expected, what came back, and where the first hit was."""
    answerable = case["source"] is not None
    return {
        "question": case["question"],
        "answerable": answerable,
        "expected_pages": case["pages"],
        "rank": first_hit_rank(hits, case) if answerable else None,
        "top_similarity": round(1 - hits[0][2], 3) if hits else None,
        "retrieved": [{"chunk_id": meta["chunk_id"], "page": meta["page"],
                       "similarity": round(1 - distance, 3)} for _, meta, distance in hits],
    }


def summarize(rows):
    """Hit rate @1, hit rate @k and mean reciprocal rank over answerable questions,
    plus the mean top similarity of answerable and unanswerable questions."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    n = len(answerable)

    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    return {
        "answerable": n,
        "unanswerable": len(unanswerable),
        "hit_rate_at_1": round(sum(r["rank"] == 1 for r in answerable) / n, 3) if n else None,
        "hit_rate_at_k": round(sum(r["rank"] is not None for r in answerable) / n, 3) if n else None,
        "mrr": round(sum(1 / r["rank"] for r in answerable if r["rank"]) / n, 3) if n else None,
        "mean_top_similarity_answerable": mean(r["top_similarity"] for r in answerable),
        "mean_top_similarity_unanswerable": mean(r["top_similarity"] for r in unanswerable),
    }


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "questions.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        sys.exit(f"No index found for the '{BACKEND}' backend. Run: python ingest.py")

    rows = []
    for case in cases:
        row = evaluate(case, retrieve(collection, case["question"]))
        rows.append(row)
        found = ", ".join(f"p{r['page']}" for r in row["retrieved"])
        if not row["answerable"]:
            mark = "no answer"
        else:
            mark = f"hit @{row['rank']}" if row["rank"] else "MISS"
        print(f"{mark:9}  top sim {row['top_similarity']:.2f}  {row['question']}"
              f"  (expected p{row['expected_pages']}, got {found})")

    summary = summarize(rows)
    print(f"\nBackend: {BACKEND}, top {TOP_K}, chunk {CHUNK_SIZE}/{CHUNK_OVERLAP}, "
          f"{summary['answerable']} answerable + {summary['unanswerable']} unanswerable questions")
    print(f"Hit rate @1:      {summary['hit_rate_at_1']:.0%}")
    print(f"Hit rate @{TOP_K}:      {summary['hit_rate_at_k']:.0%}")
    print(f"Mean recip. rank: {summary['mrr']:.2f}")
    print(f"Mean top similarity: {summary['mean_top_similarity_answerable']} (answerable), "
          f"{summary['mean_top_similarity_unanswerable']} (no answer in the documents)")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"retrieval-{BACKEND}-{datetime.now():%Y%m%d-%H%M%S}.json"
    settings = {"backend": BACKEND, "collection": COLLECTION, "top_k": TOP_K,
                "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "questions": str(path)}
    out.write_text(json.dumps({"settings": settings, "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    try:
        main()
    except LLMError as error:
        sys.exit(f"Model backend error: {error}")
