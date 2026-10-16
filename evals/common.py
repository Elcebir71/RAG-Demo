"""Shared by the evaluation scripts: load the questions, open the index, save a run."""

import json
import sys
from datetime import datetime
from pathlib import Path

import chromadb

from config import BACKEND, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DB_DIR, TOP_K

HERE = Path(__file__).parent


def load_cases(path=None):
    """The given questions file, or evals/questions.json."""
    path = Path(path) if path else HERE / "questions.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def open_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        sys.exit(f"No index found for the '{BACKEND}' backend. Run: python ingest.py")


def settings(questions_path, **extra):
    return {"backend": BACKEND, "collection": COLLECTION, "top_k": TOP_K,
            "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
            "questions": str(questions_path), **extra}


def save_run(kind, run_settings, summary, rows):
    """Write one run to evals/results/<kind>-<backend>-<time>.json and return the path."""
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"{kind}-{BACKEND}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"settings": run_settings, "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    return out
