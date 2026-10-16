"""Tests for the request trace: off by default, one line per /ask when on, never breaks an answer."""

import json

import pytest
from fastapi.testclient import TestClient

import api
import ask
import provenance

client = TestClient(api.app)


class FakeCollection:
    metadata = {"hnsw:space": "cosine", "indexed_at": "2026-10-16T21:00:00+00:00"}

    def query(self, query_embeddings, n_results):
        return {
            "ids": [["az.pdf-p13-c0", "az.pdf-p15-c1"]],
            "documents": [["IaaS gives you the most control.", "PaaS is managed."]],
            "metadatas": [[{"source": "az.pdf", "page": 13}, {"source": "az.pdf", "page": 15}]],
            "distances": [[0.35, 0.49]],
        }


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(ask, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(api, "get_collection", FakeCollection)
    monkeypatch.setattr(api, "chat", lambda system, prompt: "IaaS [1], see also [5].")


def ask_once():
    response = client.post("/ask", json={"question": "Which model gives the most control?"})
    assert response.status_code == 200


def test_nothing_is_written_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "PROVENANCE_LOG", "")
    ask_once()
    assert list(tmp_path.iterdir()) == []


def test_each_answer_adds_one_line(tmp_path, monkeypatch):
    path = tmp_path / "provenance.jsonl"
    monkeypatch.setattr(provenance, "PROVENANCE_LOG", str(path))
    ask_once()
    ask_once()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["question"] == "Which model gives the most control?"
    assert entry["indexed_at"] == "2026-10-16T21:00:00+00:00"
    assert entry["chunks"] == [{"chunk_id": "az.pdf-p13-c0", "similarity": 0.65},
                               {"chunk_id": "az.pdf-p15-c1", "similarity": 0.51}]
    assert entry["answer"] == "IaaS [1], see also [5]."
    assert (entry["cited"], entry["invalid_citations"]) == ([1], [5])
    assert entry["model"] and entry["backend"]


def test_a_failed_write_does_not_break_the_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "PROVENANCE_LOG", str(tmp_path / "missing-dir" / "log.jsonl"))
    ask_once()
