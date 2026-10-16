"""Tests for the retrieval check's scoring, with fixed hits instead of a real index."""

import json
from pathlib import Path

from evals.eval_retrieval import evaluate, first_hit_rank, summarize

PDF = "az.pdf"


def hits(*pages, source=PDF):
    """Fake retrieve() output: one chunk per page, similarity falling from 0.8."""
    return [(f"text {p}", {"source": source, "page": p, "chunk_id": f"{source}-p{p}-c0"}, 0.2 + i / 10)
            for i, p in enumerate(pages)]


def case(pages, source=PDF):
    return {"question": "q", "source": source, "pages": pages}


def test_first_hit_rank():
    assert first_hit_rank(hits(5, 13, 13), case([13])) == 2
    assert first_hit_rank(hits(5, 6), case([13])) is None
    assert first_hit_rank(hits(13, source="other.pdf"), case([13])) is None  # right page, wrong file


def test_unanswerable_question_has_no_rank_but_keeps_its_similarity():
    row = evaluate(case([], source=None), hits(13))
    assert row["answerable"] is False
    assert row["rank"] is None
    assert row["top_similarity"] == 0.8
    assert row["retrieved"] == [{"chunk_id": "az.pdf-p13-c0", "page": 13, "similarity": 0.8}]


def test_summary():
    rows = [
        evaluate(case([13]), hits(13, 5)),       # hit @1
        evaluate(case([13]), hits(5, 13)),       # hit @2
        evaluate(case([13]), hits(5, 6)),        # miss
        evaluate(case([], source=None), hits(5)),
    ]
    summary = summarize(rows)
    assert (summary["answerable"], summary["unanswerable"]) == (3, 1)
    assert summary["hit_rate_at_1"] == 0.333
    assert summary["hit_rate_at_k"] == 0.667
    assert summary["mrr"] == 0.5  # (1 + 1/2 + 0) / 3
    assert summary["mean_top_similarity_answerable"] == 0.8
    assert summary["mean_top_similarity_unanswerable"] == 0.8


def test_question_file_is_well_formed():
    cases = json.loads((Path(__file__).parent.parent / "evals" / "questions.json").read_text(encoding="utf-8"))
    for c in cases:
        if c["source"] is None:  # no answer in the documents
            assert set(c) == {"question", "source", "pages"} and c["pages"] == []
        else:
            assert set(c) == {"question", "source", "pages", "answer", "facts"}
            assert c["pages"] and c["answer"] and c["facts"]
            for fact in c["facts"]:
                assert isinstance(fact, str) or (fact and all(isinstance(a, str) for a in fact))
