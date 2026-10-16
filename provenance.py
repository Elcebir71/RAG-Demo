"""Request trace: one JSON line per /ask, so any answer can be traced back to the exact chunks.

Off by default. The API is public, so storing people's questions must be a deliberate choice:
set PROVENANCE_LOG to a file path to turn it on.
"""

import json
import logging
from datetime import datetime, timezone

from config import AZURE_CHAT_DEPLOYMENT, BACKEND, CHAT_MODEL, PROVENANCE_LOG

log = logging.getLogger("rag")


def record(collection, question, hits, answer_text, cited, invalid):
    """Append one line to PROVENANCE_LOG. A failed write is logged, never shown to the caller."""
    if not PROVENANCE_LOG:
        return
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "backend": BACKEND,
        "model": AZURE_CHAT_DEPLOYMENT if BACKEND == "azure" else CHAT_MODEL,
        # When the index was built (set by ingest.py), so an answer can be tied to an index version.
        "indexed_at": (getattr(collection, "metadata", None) or {}).get("indexed_at"),
        "chunks": [{"chunk_id": meta["chunk_id"], "similarity": round(1 - distance, 3)}
                   for _, meta, distance in hits],
        "answer": answer_text,
        "cited": sorted(cited),
        "invalid_citations": invalid,
    }
    try:
        with open(PROVENANCE_LOG, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as error:
        log.warning("Could not write the provenance log: %s", error)
