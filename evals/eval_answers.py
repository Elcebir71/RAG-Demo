"""Answer quality check: does the full pipeline cite the right sources, and say
"I don't know" when the documents have no answer?

Runs every question in questions.json through retrieval AND the chat model, then
scores the answer with plain code (no second model judges it):

  answerable questions
    citation coverage  - the answer cites at least one retrieved source as [n]
    citation validity  - every [n] in the answer points at a retrieved source
    citation accuracy  - share of cited sources that come from an expected page
    wrong abstention   - the answer says "I don't know" although the pages are there
  unanswerable questions ("source": null)
    correct abstention - the answer says "I don't know"

"I don't know" is recognised by a list of phrases in English, Dutch and Turkish,
so it can miss an unusual wording: every answer is saved, check the misses by eye.

Usage (from the repository root, needs the index and the chat model):
    python -m evals.eval_answers [questions file]
"""

import sys

from ask import SYSTEM_PROMPT, build_prompt, citations, retrieve
from config import AZURE_CHAT_DEPLOYMENT, BACKEND, CHAT_MODEL, TOP_K, USE_AZURE
from evals.common import load_cases, open_collection, save_run, settings
from llm_client import LLMError, chat

# Lower case, with ' as the only apostrophe. Add a phrase when a run shows a miss.
ABSTAIN_PHRASES = [
    # English
    "don't know", "do not know", "not in the context", "no information",
    "does not contain", "doesn't contain", "not mentioned", "cannot answer", "can't answer",
    "not provided", "no mention",
    # Dutch
    "weet ik niet", "weet het niet", "niet in de context", "geen informatie",
    "niet vermeld", "staat niet", "kan ik niet beantwoorden", "kan deze vraag niet",
    # Turkish
    "bilmiyorum", "bilgi yok", "bilgi bulunm", "yer almıyor", "yer almamaktadır",
    "bulunamadı", "bahsedilmiyor", "bahsedilmemektedir", "cevap veremiyorum", "yanıt veremiyorum",
]


def abstains(answer_text):
    """True if the answer says, in one of the known wordings, that it does not know."""
    text = answer_text.lower().replace("’", "'")
    return any(phrase in text for phrase in ABSTAIN_PHRASES)


def evaluate_answer(case, hits, answer_text):
    """One result row for a question, its retrieved chunks and the model's answer."""
    answerable = case["source"] is not None
    cited, invalid = citations(answer_text, len(hits))
    cited_meta = [hits[n - 1][1] for n in sorted(cited)]
    correct = [m for m in cited_meta
               if answerable and m["source"] == case["source"] and m["page"] in case["pages"]]
    return {
        "question": case["question"],
        "answerable": answerable,
        "expected_pages": case["pages"],
        "answer": answer_text,
        "abstained": abstains(answer_text),
        "cited": [{"n": n, "chunk_id": m["chunk_id"], "page": m["page"]}
                  for n, m in zip(sorted(cited), cited_meta)],
        "invalid_citations": invalid,
        "citation_accuracy": round(len(correct) / len(cited_meta), 3) if cited_meta and answerable else None,
    }


def summarize_answers(rows):
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def share(rows_, test):
        return round(sum(1 for r in rows_ if test(r)) / len(rows_), 3) if rows_ else None

    accuracies = [r["citation_accuracy"] for r in answerable if r["citation_accuracy"] is not None]
    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "citation_coverage": share(answerable, lambda r: r["cited"]),
        "citation_validity": share(answerable, lambda r: not r["invalid_citations"]),
        "citation_accuracy": round(sum(accuracies) / len(accuracies), 3) if accuracies else None,
        "wrong_abstention": share(answerable, lambda r: r["abstained"]),
        "correct_abstention": share(unanswerable, lambda r: r["abstained"]),
    }


def percent(value):
    return "n/a" if value is None else f"{value:.0%}"


def main():
    path, cases = load_cases()
    collection = open_collection()

    rows = []
    for case in cases:
        hits = retrieve(collection, case["question"])
        row = evaluate_answer(case, hits, chat(SYSTEM_PROMPT, build_prompt(case["question"], hits)))
        rows.append(row)
        pages = ", ".join(f"[{c['n']}] p{c['page']}" for c in row["cited"]) or "no citation"
        if row["answerable"]:
            mark = "ABSTAINED" if row["abstained"] else f"acc {percent(row['citation_accuracy'])}"
        else:
            mark = "abstained" if row["abstained"] else "ANSWERED"
        bad = f", invalid {row['invalid_citations']}" if row["invalid_citations"] else ""
        print(f"{mark:9}  {row['question']}  (expected p{row['expected_pages']}, cited {pages}{bad})")

    summary = summarize_answers(rows)
    model = AZURE_CHAT_DEPLOYMENT if USE_AZURE else CHAT_MODEL
    print(f"\nBackend: {BACKEND}, chat model {model}, top {TOP_K}, "
          f"{summary['answerable']} answerable + {summary['unanswerable']} unanswerable questions")
    print(f"Citation coverage:   {percent(summary['citation_coverage'])}  (answers that cite a source)")
    print(f"Citation validity:   {percent(summary['citation_validity'])}  (answers whose [n] all exist)")
    print(f"Citation accuracy:   {percent(summary['citation_accuracy'])}  (cited sources from an expected page)")
    print(f"Wrong abstention:    {percent(summary['wrong_abstention'])}  (said \"don't know\" though the answer is there)")
    print(f"Correct abstention:  {percent(summary['correct_abstention'])}  (said \"don't know\" when the documents have no answer)")

    out = save_run("answers", settings(path, chat_model=model), summary, rows)
    print(f"\nSaved to {out} (every answer is in there, check the misses by eye)")


if __name__ == "__main__":
    try:
        main()
    except LLMError as error:
        sys.exit(f"Model backend error: {error}")
