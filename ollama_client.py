"""Tiny wrapper around the Ollama REST API (embeddings + chat)."""

import sys

import requests

from config import CHAT_MODEL, EMBED_MODEL, OLLAMA_URL


class OllamaError(RuntimeError):
    """Raised when Ollama answers with an error (shows Ollama's own message)."""


def _post(endpoint, payload, timeout):
    try:
        response = requests.post(f"{OLLAMA_URL}{endpoint}", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        sys.exit(f"Cannot reach Ollama at {OLLAMA_URL}. Is the Ollama app running?")
    if response.status_code == 404:
        sys.exit(f"Model not found. Run: ollama pull {payload['model']}")
    if not response.ok:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        raise OllamaError(f"Ollama {response.status_code} on {endpoint}: {detail}")
    return response.json()


def embed(texts):
    """Turn a list of texts into a list of vectors."""
    data = _post("/api/embed", {"model": EMBED_MODEL, "input": texts, "truncate": True}, timeout=600)
    return data["embeddings"]


def chat(system_prompt, user_prompt):
    """Send one question to the local LLM and return its answer."""
    payload = {
        "model": CHAT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = _post("/api/chat", payload, timeout=600)
    return data["message"]["content"]