"""Tests for level filtering: a class only gets material for its school type, year and stream.

The retrieval test uses a real in-memory ChromaDB with hand-made vectors, so the filter is
checked against the database itself; no embedding model is needed.
"""

import json
import uuid

import chromadb
import pytest
from fastapi.testclient import TestClient

import api
import ask
import levels

client = TestClient(api.app)


@pytest.mark.parametrize("niveau, year, expected", [
    ("po", 1, -1), ("po", 3, 1), ("po", 8, 6),
    ("vmbo", 1, 7), ("havo", 5, 11), ("vwo", 6, 12),
])
def test_grade_puts_both_school_phases_on_one_axis(niveau, year, expected):
    assert levels.grade(niveau, year) == expected


@pytest.mark.parametrize("niveau, year", [("havo", 6), ("vmbo", 5), ("po", 9), ("vwo", 0), ("mbo", 1)])
def test_grade_rejects_years_that_do_not_exist(niveau, year):
    with pytest.raises(levels.LevelError):
        levels.grade(niveau, year)


def test_chunk_metadata_is_flat():
    meta = levels.chunk_metadata({"niveau": ["havo", "vwo"], "leerjaar": [1, 2],
                                  "domein": "Algebra", "licentie": "CC BY-SA 4.0"})
    assert meta == {"grade_min": 7, "niveau_havo": True, "niveau_vwo": True, "stroom_all": True,
                    "domein": "Algebra", "licentie": "CC BY-SA 4.0"}
    assert all(not isinstance(value, list) for value in meta.values())  # ChromaDB allows no lists


@pytest.mark.parametrize("entry", [
    {"niveau": "havo", "leerjaar": [4, 6]},           # havo ends in year 5
    {"niveau": "havo", "stroom": "C"},                # havo has no wiskunde C
    {"niveau": "vmbo", "stroom": "B"},                # vmbo has learning paths, not A/B
    {"niveau": "po", "stroom": "kb"},
    {"niveau": ["po", "vmbo"], "leerjaar": 1},        # basisschool and VO need separate files
    {"niveau": ["havo", "vwo"], "stroom": "C"},       # only vwo has C; would leak to havo
])
def test_chunk_metadata_rejects_impossible_levels(entry):
    with pytest.raises(levels.LevelError):
        levels.chunk_metadata(entry)


def test_load_manifest(tmp_path):
    assert levels.load_manifest(tmp_path) == {}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "breuken.md": {"niveau": "po", "leerjaar": [7, 8]},
        "kans.md": {"niveau": "vwo", "leerjaar": [5, 6], "stroom": "A"},
    }), encoding="utf-8")
    manifest = levels.load_manifest(tmp_path)
    assert manifest["breuken.md"]["grade_min"] == 5
    assert manifest["kans.md"]["stroom_a"] is True


def test_load_manifest_names_the_wrong_entry(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"x.md": {"niveau": "havo", "leerjaar": 6}}))
    with pytest.raises(levels.LevelError, match="x.md"):
        levels.load_manifest(tmp_path)


# --- Retrieval against a real ChromaDB ---------------------------------------------

DOCUMENTS = {
    "po-breuken": {"niveau": "po", "leerjaar": [7, 8]},
    "onderbouw-lineair": {"niveau": ["havo", "vwo"], "leerjaar": [1, 2]},
    "havo-a-statistiek": {"niveau": "havo", "leerjaar": [4, 5], "stroom": "A"},
    "havo-b-differentieren": {"niveau": "havo", "leerjaar": [4, 5], "stroom": "B"},
    "vmbo-kb-pythagoras": {"niveau": "vmbo", "leerjaar": 3, "stroom": ["kb", "gl", "tl"]},
}


@pytest.fixture
def collection(monkeypatch):
    """Every chunk is equally close to every question, so only the filter decides what comes back."""
    monkeypatch.setattr(ask, "embed", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(ask, "TOP_K", 10)
    db = chromadb.EphemeralClient().create_collection(f"test-{uuid.uuid4().hex}")
    names = [*DOCUMENTS, "azure-unlabelled"]
    db.add(ids=names, documents=names, embeddings=[[1.0, 0.0]] * len(names),
           metadatas=[{"source": name, "page": 1, **levels.chunk_metadata(entry)}
                      for name, entry in DOCUMENTS.items()] + [{"source": "azure-unlabelled", "page": 1}])
    return db


def found(collection, *level):
    where = levels.where(*level) if level else None
    return {meta["source"] for _, meta, _ in ask.retrieve(collection, "question", where=where)}


def test_without_a_level_everything_is_searched(collection):
    assert found(collection) == {*DOCUMENTS, "azure-unlabelled"}


def test_a_level_keeps_its_own_school_type_only(collection):
    assert found(collection, "po") == {"po-breuken"}
    assert found(collection, "vwo") == {"onderbouw-lineair"}


def test_higher_years_are_left_out(collection):
    assert found(collection, "havo", 3) == {"onderbouw-lineair"}
    assert found(collection, "po", 6) == set()


def test_a_stream_leaves_out_the_other_streams(collection):
    assert found(collection, "havo", 5, "a") == {"onderbouw-lineair", "havo-a-statistiek"}
    assert found(collection, "havo", 5, "B") == {"onderbouw-lineair", "havo-b-differentieren"}
    assert found(collection, "vmbo", 3, "kb") == {"vmbo-kb-pythagoras"}
    assert found(collection, "vmbo", 3, "bb") == set()


# --- API ------------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    {"question": "q", "leerjaar": 2},                       # a year without a school type
    {"question": "q", "niveau": "havo", "leerjaar": 6},
    {"question": "q", "niveau": "vmbo", "stroom": "b"},
    {"question": "q", "niveau": "mbo"},
])
def test_api_rejects_impossible_levels(body, monkeypatch):
    monkeypatch.setattr(api, "get_collection", lambda: pytest.fail("must not search"))
    assert client.post("/ask", json=body).status_code == 422


def test_api_answers_without_the_model_when_a_level_has_no_material(collection, monkeypatch):
    monkeypatch.setattr(api, "get_collection", lambda: collection)
    monkeypatch.setattr(api, "chat", lambda system, prompt: pytest.fail("must not call the model"))
    response = client.post("/ask", json={"question": "Wat is een breuk?", "niveau": "po", "leerjaar": 4})
    assert response.status_code == 200
    assert response.json() == {"answer": api.NO_MATERIAL, "sources": [], "invalid_citations": []}


def test_api_passes_the_level_to_the_search(collection, monkeypatch):
    monkeypatch.setattr(api, "get_collection", lambda: collection)
    monkeypatch.setattr(api, "chat", lambda system, prompt: "Zie [1].")
    response = client.post("/ask", json={"question": "Lineaire formules", "niveau": "vwo", "leerjaar": 2})
    assert [source["source"] for source in response.json()["sources"]] == ["onderbouw-lineair"]
