"""REST API for the RAG assistant."""

import logging

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ask import SYSTEM_PROMPT, build_prompt, retrieve
from config import BACKEND, COLLECTION, DB_DIR
from llm_client import LLMError, chat

log = logging.getLogger("rag")
app = FastAPI(title="RAG Document Assistant")


class Question(BaseModel):
    # The length limit keeps a public endpoint from being used to send huge prompts.
    question: str = Field(min_length=1, max_length=500)


class Source(BaseModel):
    source: str
    page: int
    similarity: float


class Answer(BaseModel):
    answer: str
    sources: list[Source]


def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        raise HTTPException(status_code=503, detail="No index found. Run ingest.py first.")


@app.get("/", include_in_schema=False)
def root():
    """Send visitors of the bare URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Used by Azure to check that the app is alive; also shows the active backend."""
    return {"status": "ok", "backend": BACKEND}


@app.post("/ask", response_model=Answer)
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    collection = get_collection()
    try:
        hits = retrieve(collection, body.question)
        text = chat(SYSTEM_PROMPT, build_prompt(body.question, hits))
    except LLMError as error:
        # Full detail goes to the server log; the caller only gets a generic message.
        log.error("Model backend error: %s", error)
        raise HTTPException(status_code=503, detail="The language model backend is not available.")

    sources = [
        Source(source=meta["source"], page=meta["page"], similarity=round(1 - distance, 2))
        for _, meta, distance in hits
    ]
    return Answer(answer=text, sources=sources)
