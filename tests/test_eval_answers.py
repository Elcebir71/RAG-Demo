"""Tests for the answer check's scoring, with fixed hits and answers instead of a model."""

import pytest

from evals.eval_answers import abstains, evaluate_answer, summarize_answers

PDF = "az.pdf"
HITS = [(f"text {p}", {"source": PDF, "page": p, "chunk_id": f"{PDF}-p{p}-c0"}, 0.3) for p in (13, 15, 5, 6)]
ANSWERABLE = {"question": "q", "source": PDF, "pages": [13]}
UNANSWERABLE = {"question": "q", "source": None, "pages": []}


@pytest.mark.parametrize("text", [
    "I don't know.",
    "I don’t know based on the context.",
    "The context does not contain this information.",
    "Dat weet ik niet.",
    "Bu bilgi bağlamda yer almıyor.",
    "Bilmiyorum.",
    "Bu bilgi veri kaynaklarında bulunamadı.",
])
def test_abstains(text):
    assert abstains(text)


def test_a_real_answer_does_not_count_as_abstaining():
    assert not abstains("The three models are IaaS, PaaS and SaaS [1].")


def test_answer_citing_the_right_and_a_wrong_page():
    row = evaluate_answer(ANSWERABLE, HITS, "IaaS [1], also see [3].")
    assert [c["page"] for c in row["cited"]] == [13, 5]
    assert row["citation_accuracy"] == 0.5
    assert row["invalid_citations"] == []
    assert row["abstained"] is False


def test_answer_without_citations_has_no_accuracy():
    row = evaluate_answer(ANSWERABLE, HITS, "IaaS, PaaS and SaaS.")
    assert row["cited"] == []
    assert row["citation_accuracy"] is None


def test_unanswerable_question():
    row = evaluate_answer(UNANSWERABLE, HITS, "I don't know.")
    assert row["abstained"] is True
    assert row["citation_accuracy"] is None


def test_summary():
    rows = [
        evaluate_answer(ANSWERABLE, HITS, "IaaS [1]."),             # cited, accurate
        evaluate_answer(ANSWERABLE, HITS, "IaaS [1] and [3] [9]."),  # half accurate, invalid [9]
        evaluate_answer(ANSWERABLE, HITS, "I don't know."),          # wrong abstention, no citation
        evaluate_answer(UNANSWERABLE, HITS, "Bilmiyorum."),          # correct abstention
        evaluate_answer(UNANSWERABLE, HITS, "Paris [2]."),           # answered anyway
    ]
    summary = summarize_answers(rows)
    assert (summary["answerable"], summary["unanswerable"]) == (3, 2)
    assert summary["citation_coverage"] == 0.667
    assert summary["citation_validity"] == 0.667
    assert summary["citation_accuracy"] == 0.75  # mean of 1.0 and 0.5
    assert summary["wrong_abstention"] == 0.333
    assert summary["correct_abstention"] == 0.5
