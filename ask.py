"""Steps 2 + 3 - Retrieval and generation: find relevant chunks, let the LLM answer."""

import re
import sys

import chromadb

from config import BACKEND, COLLECTION, DB_DIR, TOP_K
from llm_client import LLMError, chat, embed

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "If the answer is not in the context, say that you don't know. "
    "Cite the sources you used with their numbers, like [1] or [2]. "
    "Answer in the same language as the question."
)

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [2], and also [1, 3]
SNIPPET_LENGTH = 200

def retrieve(collection, question, where=None):
    """Return the TOP_K most similar chunks with their metadata and distance.

    The chunk id (like "file.pdf-p13-c2") is added to the metadata as "chunk_id",
    so an answer can be traced back to the exact chunk, not only the page.
    `where` is a ChromaDB metadata filter, like levels.where("havo", 2), to search one level only.
    """
    level = {"where": where} if where else {}
    result = collection.query(query_embeddings=embed([question]), n_results=TOP_K, **level)
    metadatas = [{**meta, "chunk_id": chunk_id}
                 for chunk_id, meta in zip(result["ids"][0], result["metadatas"][0])]
    return list(zip(result["documents"][0], metadatas, result["distances"][0]))


def citations(answer_text, hit_count):
    """Return (cited, invalid): the source numbers the answer cites that exist, and those that do not.

    The model is asked to cite [1]..[hit_count]; any other number points at nothing.
    """
    numbers = {int(n) for group in CITATION.findall(answer_text) for n in group.split(",")}
    cited = {n for n in numbers if 1 <= n <= hit_count}
    return cited, sorted(numbers - cited)


def snippet(text, length=SNIPPET_LENGTH):
    """The start of a chunk, cut at a space, to show where an answer came from."""
    if len(text) <= length:
        return text
    cut = text.rfind(" ", 0, length)
    return text[:cut if cut > 0 else length] + " ..."


def build_prompt(question, hits):
    context = "\n\n".join(
        f"[{n}] ({meta['source']}, page {meta['page']})\n{text}"
        for n, (text, meta, _) in enumerate(hits, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(collection, question):
    hits = retrieve(collection, question)
    text = chat(SYSTEM_PROMPT, build_prompt(question, hits))
    cited, invalid = citations(text, len(hits))
    print("\n" + text)
    print("\nSources (* = cited in the answer):")
    for n, (_, meta, distance) in enumerate(hits, start=1):
        mark = "*" if n in cited else " "
        print(f" {mark}[{n}] {meta['source']}, page {meta['page']}  "
              f"(similarity {1 - distance:.2f}, chunk {meta['chunk_id']})")
    if not cited:
        print("Warning: the answer cites none of the sources.")
    if invalid:
        print(f"Warning: the answer cites sources that do not exist: {invalid}")


def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        sys.exit(f"No index found for the '{BACKEND}' backend. Run: python ingest.py")

    if len(sys.argv) > 1:  # one-shot: python ask.py "your question"
        answer(collection, " ".join(sys.argv[1:]))
        return

    print(f"Ask a question about your documents (backend: {BACKEND}, empty line to quit).")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        answer(collection, question)


if __name__ == "__main__":
    try:
        main()
    except LLMError as error:
        sys.exit(f"Model backend error: {error}")
