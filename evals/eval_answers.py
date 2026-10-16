"""Answer quality check: is the answer right, does it cite the right sources, and
does the model say "I don't know" when the documents have no answer?

Runs every question in questions.json through retrieval AND the chat model, then
scores the answer with plain code (a model judges it only with --judge):

  answerable questions
    fact recall        - share of the key facts in questions.json found in the answer
    complete answers   - answers that contain every key fact
    citation coverage  - the answer cites at least one retrieved source as [n]
    citation validity  - every [n] in the answer points at a retrieved source
    citation accuracy  - share of cited sources that come from an expected page
    wrong abstention   - the answer says "I don't know" although the pages are there
  unanswerable questions ("source": null)
    correct abstention - the answer says "I don't know"

"I don't know" is recognised by a list of phrases in English, Dutch and Turkish,
so it can miss an unusual wording: every answer is saved, check the misses by eye.

Key facts are short, mostly language-independent strings (a number, a product name),
each with alternatives, found anywhere in the answer: "veri" also finds "verilerini" and
"crediet" finds "dienstcrediet". Numbers must stand alone, so "72" does not match "720".
They catch a wrong number or a missing item, not a wrong explanation. For that, --judge
asks the chat model to compare each answer with the reference answer in questions.json
and say correct, partial or wrong. The model then grades its own kind of answers, so
treat the verdicts as a second opinion, not as ground truth.

Usage (from the repository root, needs the index and the chat model):
    python -m evals.eval_answers [--judge] [questions file]
"""

import argparse
import json
import re
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


JUDGE_PROMPT = (
    "You grade answers of a question-answering system against a reference answer. "
    "The answer may be in another language than the reference; judge the meaning only. "
    "Check each fact against the answer text itself; do not assume a fact is there. "
    "correct: it states the essential facts of the reference and nothing that contradicts it. "
    "Extra information that does not contradict the reference is fine. "
    "partial: it is right but misses part of the reference. "
    "wrong: it contradicts the reference, or does not answer the question. "
    'Reply with JSON only: {"verdict": "correct" | "partial" | "wrong", "reason": "<one sentence>"}'
)
VERDICTS = ("correct", "partial", "wrong")


def _normalize(text):
    # casefold() turns the Turkish "İ" into "i" plus a combining dot; drop the dot.
    return text.casefold().replace("\u0307", "").replace("\u2019", "'")


def has_fact(answer_text, fact):
    """True if any alternative of the fact appears in the answer (a number only as a whole number)."""
    text = _normalize(answer_text)
    for alternative in [fact] if isinstance(fact, str) else fact:
        alternative = _normalize(alternative)
        if alternative.isdigit():
            if re.search(r"(?<!\d)" + alternative + r"(?!\d)", text):
                return True
        elif alternative in text:
            return True
    return False


def judge(case, answer_text, ask=None):
    """Ask the model whether the answer matches the reference. Returns (verdict, reason)."""
    prompt = (f"Question: {case['question']}\n\nReference answer: {case['answer']}\n\n"
              f"Answer to grade: {answer_text}")
    reply = (ask or chat)(JUDGE_PROMPT, prompt)
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    try:
        result = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        result = {}
    verdict = str(result.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return "unparsed", reply[:300]
    return verdict, str(result.get("reason", ""))


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
    facts = case.get("facts", []) if answerable else []
    missing = [f for f in facts if not has_fact(answer_text, f)]
    return {
        "question": case["question"],
        "answerable": answerable,
        "expected_pages": case["pages"],
        "reference": case.get("answer"),
        "answer": answer_text,
        "fact_recall": round(1 - len(missing) / len(facts), 3) if facts else None,
        "missing_facts": missing,
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
    with_facts = [r for r in answerable if r["fact_recall"] is not None]
    judged = [r for r in answerable if r.get("verdict")]
    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "fact_recall": round(sum(r["fact_recall"] for r in with_facts) / len(with_facts), 3) if with_facts else None,
        "complete_answers": share(with_facts, lambda r: not r["missing_facts"]),
        "judged_correct": share(judged, lambda r: r["verdict"] == "correct"),
        "judged_partial": share(judged, lambda r: r["verdict"] == "partial"),
        "judged_wrong": share(judged, lambda r: r["verdict"] == "wrong"),
        "citation_coverage": share(answerable, lambda r: r["cited"]),
        "citation_validity": share(answerable, lambda r: not r["invalid_citations"]),
        "citation_accuracy": round(sum(accuracies) / len(accuracies), 3) if accuracies else None,
        "wrong_abstention": share(answerable, lambda r: r["abstained"]),
        "correct_abstention": share(unanswerable, lambda r: r["abstained"]),
    }


def percent(value):
    return "n/a" if value is None else f"{value:.0%}"


def main():
    parser = argparse.ArgumentParser(description="Score the answers of the full RAG pipeline.")
    parser.add_argument("questions", nargs="?", help="questions file (default: evals/questions.json)")
    parser.add_argument("--judge", action="store_true",
                        help="also let the chat model compare each answer with the reference answer")
    args = parser.parse_args()
    path, cases = load_cases(args.questions)
    collection = open_collection()

    rows = []
    for case in cases:
        hits = retrieve(collection, case["question"])
        row = evaluate_answer(case, hits, chat(SYSTEM_PROMPT, build_prompt(case["question"], hits)))
        if args.judge and row["answerable"] and case.get("answer"):
            row["verdict"], row["verdict_reason"] = judge(case, row["answer"])
        rows.append(row)
        pages = ", ".join(f"[{c['n']}] p{c['page']}" for c in row["cited"]) or "no citation"
        if row["answerable"]:
            mark = "ABSTAINED" if row["abstained"] else f"facts {percent(row['fact_recall'])}"
        else:
            mark = "abstained" if row["abstained"] else "ANSWERED"
        bad = f", invalid {row['invalid_citations']}" if row["invalid_citations"] else ""
        verdict = f", judge: {row['verdict']}" if row.get("verdict") else ""
        print(f"{mark:10} {row['question']}  (cited {pages}{bad}{verdict})")
        for fact in row["missing_facts"]:
            print(f"{'':10}   missing: {fact if isinstance(fact, str) else ' / '.join(fact)}")

    summary = summarize_answers(rows)
    model = AZURE_CHAT_DEPLOYMENT if USE_AZURE else CHAT_MODEL
    print(f"\nBackend: {BACKEND}, chat model {model}, top {TOP_K}, "
          f"{summary['answerable']} answerable + {summary['unanswerable']} unanswerable questions")
    print(f"Fact recall:         {percent(summary['fact_recall'])}  (key facts found in the answer)")
    print(f"Complete answers:    {percent(summary['complete_answers'])}  (answers with every key fact)")
    if args.judge:
        print(f"Judge:               {percent(summary['judged_correct'])} correct, "
              f"{percent(summary['judged_partial'])} partial, {percent(summary['judged_wrong'])} wrong")
    print(f"Citation coverage:   {percent(summary['citation_coverage'])}  (answers that cite a source)")
    print(f"Citation validity:   {percent(summary['citation_validity'])}  (answers whose [n] all exist)")
    print(f"Citation accuracy:   {percent(summary['citation_accuracy'])}  (cited sources from an expected page)")
    print(f"Wrong abstention:    {percent(summary['wrong_abstention'])}  (said \"don't know\" though the answer is there)")
    print(f"Correct abstention:  {percent(summary['correct_abstention'])}  (said \"don't know\" when the documents have no answer)")

    out = save_run("answers", settings(path, chat_model=model, judge=args.judge), summary, rows)
    print(f"\nSaved to {out} (every answer is in there, check the misses by eye)")


if __name__ == "__main__":
    try:
        main()
    except LLMError as error:
        sys.exit(f"Model backend error: {error}")
