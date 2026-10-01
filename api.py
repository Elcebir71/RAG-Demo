import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ask import SYSTEM_PROMPT, build_prompt, retrieve
from config import COLLECTION, DB_DIR
from ollama_client import OllamaError, chat

app = FastAPI(title="Local RAF Document Assistant")

class Question(BaseModel):
    question: str

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
        raise HTTPException(STATUS_CODE=503, detail="No index found. Run ingest.py first.")

@app.get("/health")
def health():
    """Used later by Azure to check that the app is alive."""
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    collection = get_collection()
    try:
        hits = retrieve(collection, body.question)
        text = chat(SYSTEM_PROMPT, build_prompt(body.question, hits))
    except (OllamaError, SystemExit):
        raise HTTPException(status_code=503, detail="Language model backend is not reachable.")

    sources = [
        Source(source=meta["source"], page=meta["page"], similarity=round(1 - distance, 2))
        for _, meta, distance in hits
    ]

    return Answer(answer=text, sources=sources)