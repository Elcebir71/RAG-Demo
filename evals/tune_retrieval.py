"""Try several chunk sizes and top-k values, and check whether a similarity cutoff would work.

For each chunk size the documents are chunked and embedded into a throwaway in-memory
index, so the real index in db/ is never touched. Each question is embedded once and
asked against every index; top-k values are compared by cutting the same ranked list.
Only the embedding model is called, never the chat model.

Usage (from the repository root):
    python -m evals.tune_retrieval [--chunk-sizes 400,800,1200] [--top-k 2,3,4,6,8]
Results are printed and saved to evals/results/ so runs can be compared.
"""

import argparse
import sys
import time

import chromadb

from config import BACKEND, CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, TOP_K
from evals.common import load_cases, save_run
from evals.eval_retrieval import evaluate, summarize
from ingest import BATCH_SIZE, chunk_text, read_documents
from llm_client import LLMError, embed


def build_index(pages, size, overlap):
    """Chunk and embed the pages into a new in-memory collection. Returns (collection, chunk count)."""
    ids, documents, metadatas = [], [], []
    for source, page, text in pages:
        for i, chunk in enumerate(chunk_text(text, size, overlap)):
            ids.append(f"{source}-p{page}-c{i}")
            documents.append(chunk)
            metadatas.append({"source": source, "page": page})

    client = chromadb.EphemeralClient()
    name = f"tune-{size}-{overlap}"
    try:
        client.delete_collection(name)  # the in-memory client is shared within one process
    except Exception:
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(documents), BATCH_SIZE):
        batch = slice(start, start + BATCH_SIZE)
        collection.add(ids=ids[batch], documents=documents[batch],
                       embeddings=embed(documents[batch]), metadatas=metadatas[batch])
    return collection, len(documents)


def ranked_hits(collection, vector, k):
    """The k closest chunks in the shape ask.retrieve() returns."""
    result = collection.query(query_embeddings=[vector], n_results=k)
    metadatas = [{**meta, "chunk_id": chunk_id}
                 for chunk_id, meta in zip(result["ids"][0], result["metadatas"][0])]
    return list(zip(result["documents"][0], metadatas, result["distances"][0]))


def cutoff_report(rows):
    """Could a minimum similarity on the best chunk tell answerable from unanswerable questions?

    Returns the lowest top similarity of an answerable question, the highest of an
    unanswerable one, and a cutoff halfway between them if they do not overlap.
    """
    answerable = [r["top_similarity"] for r in rows if r["answerable"]]
    unanswerable = [r["top_similarity"] for r in rows if not r["answerable"]]
    if not answerable or not unanswerable:
        return {"lowest_answerable": None, "highest_unanswerable": None, "gap": None, "cutoff": None}
    low, high = min(answerable), max(unanswerable)
    return {
        "lowest_answerable": low,
        "highest_unanswerable": high,
        "gap": round(low - high, 3),
        "cutoff": round((low + high) / 2, 3) if low > high else None,
    }


def sweep(cases, hits_per_case, top_ks):
    """Summaries for each top-k, cutting the same ranked hits per question."""
    return {k: summarize([evaluate(case, hits[:k]) for case, hits in zip(cases, hits_per_case)])
            for k in top_ks}


def numbers(text):
    return sorted({int(n) for n in text.split(",") if n.strip()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="?", help="questions file (default: evals/questions.json)")
    parser.add_argument("--chunk-sizes", type=numbers, default=[400, 600, 800, 1200])
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="characters shared between chunks")
    parser.add_argument("--top-k", type=numbers, default=[2, 3, 4, 6, 8])
    args = parser.parse_args()

    path, cases = load_cases(args.questions)
    pages = list(read_documents(DOCS_DIR))
    if not pages:
        sys.exit(f"No documents found in '{DOCS_DIR}/'.")
    vectors = embed([case["question"] for case in cases])
    max_k = max(args.top_k)

    print(f"Backend: {BACKEND}, current setting: chunk {CHUNK_SIZE}/{CHUNK_OVERLAP}, top {TOP_K}\n")
    print(f"{'chunk':>6} {'chunks':>6} {'top k':>5} {'hit@1':>6} {'hit@k':>6} {'MRR':>5}")
    runs = []
    for size in args.chunk_sizes:
        started = time.time()
        collection, count = build_index(pages, size, args.overlap)
        hits_per_case = [ranked_hits(collection, vector, max_k) for vector in vectors]
        summaries = sweep(cases, hits_per_case, args.top_k)
        cutoff = cutoff_report([evaluate(case, hits) for case, hits in zip(cases, hits_per_case)])
        for k, summary in summaries.items():
            print(f"{size:>6} {count:>6} {k:>5} {summary['hit_rate_at_1']:>6.0%} "
                  f"{summary['hit_rate_at_k']:>6.0%} {summary['mrr']:>5.2f}")
        verdict = f"cutoff {cutoff['cutoff']} would work" if cutoff["cutoff"] else "no clean cutoff"
        print(f"{'':>6} lowest answerable {cutoff['lowest_answerable']}, highest unanswerable "
              f"{cutoff['highest_unanswerable']}: {verdict}  ({time.time() - started:.0f}s)\n")
        runs.append({"chunk_size": size, "chunk_overlap": args.overlap, "chunks": count,
                     "by_top_k": {str(k): s for k, s in summaries.items()}, "cutoff": cutoff})

    run_settings = {"backend": BACKEND, "questions": str(path), "chunk_sizes": args.chunk_sizes,
                    "chunk_overlap": args.overlap, "top_ks": args.top_k}
    print(f"Saved to {save_run('tuning', run_settings, None, runs)}")


if __name__ == "__main__":
    try:
        main()
    except LLMError as error:
        sys.exit(f"Model backend error: {error}")
