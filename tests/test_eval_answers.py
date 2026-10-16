"""Tests for the answer check's scoring, with fixed hits and answers instead of a model."""

import pytest

from evals.eval_answers import abstains, evaluate_answer, has_fact, judge, summarize_answers

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


def case_with(facts, answer="reference"):
    return {**ANSWERABLE, "answer": answer, "facts": facts}


@pytest.mark.parametrize("text, fact, found", [
    ("The passing score is 700.", "700", True),
    ("Kosten dalen tot 720 euro.", "72", False),         # a number must stand alone
    ("Tot %72 korting.", "72", True),
    ("Müşteri verilerini korur.", ["veri", "data"], True),  # Turkish suffix
    ("Je krijgt een dienstcrediet.", ["credit", "crediet"], True),  # Dutch compound
    ("İSTANBUL", "istanbul", True),                      # Turkish capital İ
    ("IaaS and PaaS.", "SaaS", False),
])
def test_has_fact(text, fact, found):
    assert has_fact(text, fact) is found


def test_missing_facts_are_listed():
    row = evaluate_answer(case_with(["IaaS", "PaaS", "SaaS"]), HITS, "IaaS and PaaS [1].")
    assert row["fact_recall"] == 0.667
    assert row["missing_facts"] == ["SaaS"]


@pytest.mark.parametrize("reply, verdict", [
    ('{"verdict": "correct", "reason": "Same facts."}', "correct"),
    ('Sure! ```json\n{"verdict": "Wrong", "reason": "Says 90%, not 72%."}\n```', "wrong"),
    ("I think it is partially right.", "unparsed"),
    ('{"verdict": "maybe"}', "unparsed"),
])
def test_judge_reads_the_verdict(reply, verdict):
    assert judge(case_with(["72"]), "Up to 90%.", ask=lambda system, prompt: reply)[0] == verdict


def test_judge_sees_question_reference_and_answer():
    seen = {}

    def ask(system, prompt):
        seen["prompt"] = prompt
        return '{"verdict": "partial", "reason": "x"}'

    judge(case_with(["72"], answer="Up to 72%."), "Up to 90%.", ask=ask)
    assert "Up to 72%." in seen["prompt"] and "Up to 90%." in seen["prompt"]


def test_summary_counts_facts_and_verdicts():
    rows = [
        {**evaluate_answer(case_with(["700"]), HITS, "700 [1]."), "verdict": "correct"},
        {**evaluate_answer(case_with(["72"]), HITS, "90% [1]."), "verdict": "wrong"},
    ]
    summary = summarize_answers(rows)
    assert summary["fact_recall"] == 0.5
    assert summary["complete_answers"] == 0.5
    assert (summary["judged_correct"], summary["judged_partial"], summary["judged_wrong"]) == (0.5, 0.0, 0.5)
