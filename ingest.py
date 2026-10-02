"""Step 1 - Indexing: read documents, split into chunks, embed, store in ChromaDB."""

import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader

from config import BACKEND, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DB_DIR, DOCS_DIR
from llm_client import LLMError, LLMUnavailable, embed

BATCH_SIZE = 8  # small batches are easier on RAM / GPU memory and on rate limits


def embed_safely(texts, ids):
    """Embed a batch; if it fails, retry one by one and skip only the failing chunks."""
    try:
        return list(zip(ids, texts, embed(texts)))
    except LLMUnavailable:
        raise  # nothing will work; stop instead of failing chunk by chunk
    except LLMError as error:
        print(f"  Batch failed ({error}). Retrying chunks one by one...")
    results = []
    for chunk_id, text in zip(ids, texts):
        try:
            results.append((chunk_id, text, embed([text])[0]))
        except LLMUnavailable:
            raise
        except LLMError as error:
            print(f"  Skipped {chunk_id}: {error}")
    return results


def read_documents(folder):
    """Yield (file name, page number, text) for every page / file."""
    for path in sorted(Path(folder).iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    yield path.name, page_number, text
        elif suffix in (".txt", ".md"):
            yield path.name, 1, path.read_text(encoding="utf-8")


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring to cut at whitespace."""
    text = " ".join(text.split())  # normalise whitespace
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def main():
    ids, documents, metadatas = [], [], []
    for source, page, text in read_documents(DOCS_DIR):
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}-p{page}-c{i}")
            documents.append(chunk)
            metadatas.append({"source": source, "page": page})

    if not documents:
        print(f"No documents found in '{DOCS_DIR}/'. Add some PDF, TXT or MD files first.")
        return

    print(f"Indexing {len(documents)} chunks with the '{BACKEND}' backend into '{COLLECTION}'")
    embed(documents[:1])  # fail fast, before the old index is deleted

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)  # rebuild from scratch each run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    meta_by_id = dict(zip(ids, metadatas))
    stored = 0
    for start in range(0, len(documents), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        results = embed_safely(documents[batch], ids[batch])
        if results:
            batch_ids, batch_docs, batch_vectors = map(list, zip(*results))
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_vectors,
                metadatas=[meta_by_id[i] for i in batch_ids],
            )
            stored += len(results)
        print(f"Processed {min(start + BATCH_SIZE, len(documents))}/{len(documents)} chunks")

    print(f"Done. Stored {stored}/{len(documents)} chunks. Now run: python ask.py")


if __name__ == "__main__":
    try:
        main()
    except LLMError as error:
        sys.exit(f"Model backend error: {error}")
