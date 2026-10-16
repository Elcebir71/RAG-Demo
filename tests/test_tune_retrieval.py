"""Tests for the tuning sweep: top-k cuts, the cutoff check and the throwaway index.

A fake embedding stands in for the model, so no backend is needed.
"""

from evals import tune_retrieval as tune

SOURCE = "az.pdf"


def row(answerable, top_similarity):
    return {"answerable": answerable, "top_similarity": top_similarity}


def test_cutoff_is_proposed_only_when_the_groups_do_not_overlap():
    apart = [row(True, 0.62), row(True, 0.55), row(False, 0.41), row(False, 0.45)]
    assert tune.cutoff_report(apart) == {
        "lowest_answerable": 0.55, "highest_unanswerable": 0.45, "gap": 0.1, "cutoff": 0.5}

    overlapping = [row(True, 0.48), row(False, 0.5)]
    report = tune.cutoff_report(overlapping)
    assert report["gap"] == -0.02 and report["cutoff"] is None


def test_cutoff_needs_both_kinds_of_question():
    assert tune.cutoff_report([row(True, 0.6)])["cutoff"] is None


def hit(page, distance):
    return ("text", {"source": SOURCE, "page": page, "chunk_id": f"{SOURCE}-p{page}-c0"}, distance)


def test_sweep_cuts_the_same_ranking_at_each_top_k():
    cases = [{"question": "q", "source": SOURCE, "pages": [3]}]
    hits = [[hit(1, 0.3), hit(2, 0.4), hit(3, 0.5)]]
    summaries = tune.sweep(cases, hits, [1, 2, 3])
    assert [summaries[k]["hit_rate_at_k"] for k in (1, 2, 3)] == [0, 0, 1]
    assert summaries[3]["mrr"] == round(1 / 3, 3)


def fake_embed(texts):
    """Letter counts: texts that share words end up close together."""
    return [[text.lower().count(letter) + 0.01 for letter in "abcdefghijklmnopqrstuvwxyz"]
            for text in texts]


def test_build_index_chunks_and_retrieves_by_page(monkeypatch):
    monkeypatch.setattr(tune, "embed", fake_embed)
    pages = [(SOURCE, 1, "zebra zebra zebra " * 30), (SOURCE, 2, "apple banana apple " * 30)]
    collection, count = tune.build_index(pages, 200, 50)
    assert count > 2

    [vector] = fake_embed(["banana apple"])
    [(_, meta, _)] = tune.ranked_hits(collection, vector, 1)
    assert meta["page"] == 2 and meta["chunk_id"].startswith(f"{SOURCE}-p2-")


def test_numbers_parses_a_comma_list():
    assert tune.numbers("8, 2,4,4") == [2, 4, 8]
