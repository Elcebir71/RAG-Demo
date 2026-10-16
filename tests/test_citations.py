"""Tests for source tracking: chunk ids, snippets and checking the answer's [n] citations.

A fake collection and a fake model stand in for ChromaDB and the LLM, so no backend is needed.
"""

import pytest
from fastapi.testclient import TestClient

import api
import ask

client = TestClient(api.app)


class FakeCollection:
    """Returns two fixed chunks for any question, in the shape ChromaDB's query() uses."""

    def query(self, query_embeddings, n_results):
        return {
            "ids": [["az.pdf-p13-c0", "az.pdf-p15-c1"]],
            "documents": [["IaaS gives you the most control. " * 10, "PaaS is managed."]],
            "metadatas": [[{"source": "az.pdf", "page": 13}, {"source": "az.pdf", "page": 15}]],
            "distances": [[0.35, 0.49]],
        }


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr(ask, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(api, "get_collection", FakeCollection)

    def answer_with(text):
        monkeypatch.setattr(api, "chat", lambda system, prompt: text)

    return answer_with


@pytest.mark.parametrize("text, cited, invalid", [
    ("IaaS [1] and PaaS [2].", {1, 2}, []),
    ("Both [1][2].", {1, 2}, []),
    ("Both [1, 2].", {1, 2}, []),
    ("See [2] and [7].", {2}, [7]),
    ("See [0].", set(), [0]),
    ("No citation at all.", set(), []),
    ("A list [a] is not a citation.", set(), []),
])
def test_citations(text, cited, invalid):
    assert ask.citations(text, 4) == (cited, invalid)


def test_snippet_is_short_and_cut_at_a_space():
    text = "word " * 100
    short = ask.snippet(text)
    assert len(short) <= ask.SNIPPET_LENGTH + 4
    assert short.endswith("word ...")
    assert ask.snippet("short text") == "short text"


def test_retrieve_adds_the_chunk_id(monkeypatch):
    monkeypatch.setattr(ask, "embed", lambda texts: [[0.0]])
    [(_, meta, _), _] = ask.retrieve(FakeCollection(), "question")
    assert meta == {"source": "az.pdf", "page": 13, "chunk_id": "az.pdf-p13-c0"}


def test_ask_marks_which_sources_the_answer_cites(fake_backend):
    fake_backend("IaaS gives the most control [1].")
    body = client.post("/ask", json={"question": "Which model gives the most control?"}).json()

    assert body["invalid_citations"] == []
    first, second = body["sources"]
    assert (first["chunk_id"], first["cited"]) == ("az.pdf-p13-c0", True)
    assert (second["chunk_id"], second["cited"]) == ("az.pdf-p15-c1", False)
    assert first["snippet"].startswith("IaaS gives you the most control.")
    assert len(first["snippet"]) < 220
    assert first["similarity"] == 0.65


def test_ask_reports_citations_that_point_at_nothing(fake_backend):
    fake_backend("PaaS is managed [2], see also [5].")
    body = client.post("/ask", json={"question": "What is PaaS?"}).json()

    assert body["invalid_citations"] == [5]
    assert [s["cited"] for s in body["sources"]] == [False, True]
