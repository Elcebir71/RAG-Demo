"""REST API for the RAG assistant and the agent."""

import json
import logging
import secrets
from typing import Any, Literal

import chromadb
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import agent
import store
from ask import SYSTEM_PROMPT, build_prompt, retrieve
from config import ADMIN_API_KEY, BACKEND, COLLECTION, DB_DIR
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
# --- Agent ----------------------------------------------------------------------

Status = Literal["pending", "rejected", "running", "done", "failed"]


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class ActionOut(BaseModel):
    id: int
    tool: str
    args: Any  # shown so a reviewer sees exactly what they approve
    risk: str | None
    status: Status
    detail: str | None


class AgentReply(BaseModel):
    answer: str
    actions: list[ActionOut]


def _out(action):
    return {**action, "args": json.loads(action["args"])}


def require_admin(x_admin_key: str = Header(default="")):
    """Approval is a human decision, so it needs a key that neither the model nor the public has."""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Approvals are disabled: ADMIN_API_KEY is not set.")
    if not secrets.compare_digest(x_admin_key.encode(), ADMIN_API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid admin key.")


@app.post("/agent", response_model=AgentReply, dependencies=[Depends(require_admin)])
def run_agent(body: AgentRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty.")
    try:
        result = agent.run_agent(body.message)
    except LLMError as error:
        log.error("Model backend error: %s", error)
        raise HTTPException(status_code=503, detail="The language model backend is not available.")
    return {"answer": result["answer"], "actions": [_out(a) for a in result["actions"]]}


@app.get("/actions", response_model=list[ActionOut], dependencies=[Depends(require_admin)])
def list_actions(status: Status | None = None):
    return [_out(a) for a in store.list_actions(status)]


@app.post("/actions/{action_id}/approve", response_model=ActionOut, dependencies=[Depends(require_admin)])
def approve_action(action_id: int):
    try:
        return _out(agent.approve(action_id))
    except LookupError:
        raise HTTPException(status_code=404, detail="No such action.")


@app.post("/actions/{action_id}/reject", response_model=ActionOut, dependencies=[Depends(require_admin)])
def reject_action(action_id: int):
    try:
        return _out(agent.reject(action_id))
    except LookupError:
        raise HTTPException(status_code=404, detail="No such action.")


@app.get("/actions/{action_id}/audit", dependencies=[Depends(require_admin)])
def action_audit(action_id: int):
    return store.audit_log(action_id)