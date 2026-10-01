"""Steps 2 + 3 - Retrieval and generation: find relevant chunks, let the LLM answer."""

import sys

import chromadb

from config import COLLECTION, DB_DIR, TOP_K
from ollama_client import chat, embed

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "If the answer is not in the context, say that you don't know. "
    "Cite the sources you used with their numbers, like [1] or [2]. "
    "Answer in the same language as the question."
)


def retrieve(collection, question):
    """Return the TOP_K most similar chunks with their metadata and distance."""
    result = collection.query(query_embeddings=embed([question]), n_results=TOP_K)
    return list(zip(result["documents"][0], result["metadatas"][0], result["distances"][0]))


def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{n}] ({meta['source']}, page {meta['page']})\n{text}"
        for n, (text, meta, _) in enumerate(hits, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(collection, body):
    hits = retrieve(collection)
    print("\n" + chat(SYSTEM_PROMPT, build_prompt(body.question, hits)))
    print("\nSources:")
    for n, (_, meta, distance) in enumerate(hits, start=1):
        print(f"  [{n}] {meta['source']}, page {meta['page']}  (similarity {1 - distance:.2f})")


def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        sys.exit("No index found. Run: python ingest.py")

    if len(sys.argv) > 1:  # one-shot: python ask.py "your question"
        class Body:
            def __init__(self, question):
                self.question = question
        answer(collection, Body(" ".join(sys.argv[1:])))
        return

    print("Ask a question about your documents (empty line to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        answer(collection, question)


if __name__ == "__main__":
    main()
