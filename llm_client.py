"""Model backend: Azure OpenAI in the cloud, Ollama on a local machine.

Which one is used is decided by environment variables (see config.py), so the
same code runs unchanged on a laptop and in Azure.
"""

import time

import requests

from config import (
    AZURE_API_KEY,
    AZURE_CHAT_DEPLOYMENT,
    AZURE_EMBED_DEPLOYMENT,
    AZURE_ENDPOINT,
    AZURE_REASONING_EFFORT,
    CHAT_MODEL,
    EMBED_MODEL,
    OLLAMA_URL,
    USE_AZURE,
)


class LLMError(RuntimeError):
    """A single request to the model backend failed."""


class LLMUnavailable(LLMError):
    """The backend cannot be used at all (unreachable, bad key, missing model)."""


def _detail(response):
    """Extract the backend's own error message from a failed response."""
    try:
        error = response.json().get("error", response.text)
    except ValueError:
        return response.text
    return error.get("message", str(error)) if isinstance(error, dict) else str(error)


def _post(url, payload, headers=None, retries=0, not_found=""):
    """POST JSON and return the parsed reply. Waits and retries when rate limited (429)."""
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=600)
        except requests.exceptions.RequestException as error:
            raise LLMUnavailable(f"Cannot reach the model backend: {error}") from error
        if response.status_code != 429 or attempt == retries:
            break
        try:
            wait = min(float(response.headers.get("Retry-After", 10)), 60)
        except ValueError:
            wait = 10
        print(f"  Rate limited, waiting {wait:.0f}s...", flush=True)
        time.sleep(wait)

    if response.status_code in (401, 403):
        raise LLMUnavailable(f"Access denied ({response.status_code}). Check the API key. {_detail(response)}")
    if response.status_code == 404:
        raise LLMUnavailable(not_found or _detail(response))
    if not response.ok:
        raise LLMError(f"{response.status_code}: {_detail(response)}")
    return response.json()


def _azure(path, payload, retries):
    if not AZURE_API_KEY:
        raise LLMUnavailable("AZURE_OPENAI_API_KEY is not set.")
    return _post(
        f"{AZURE_ENDPOINT}/openai/v1/{path}",
        payload,
        headers={"api-key": AZURE_API_KEY},
        retries=retries,
        not_found=f"Azure deployment '{payload['model']}' not found at {AZURE_ENDPOINT}.",
    )


def _ollama(path, payload):
    return _post(
        f"{OLLAMA_URL}{path}",
        payload,
        not_found=f"Model not found. Run: ollama pull {payload['model']}",
    )


def embed(texts):
    """Turn a list of texts into a list of vectors."""
    if USE_AZURE:
        data = _azure("embeddings", {"model": AZURE_EMBED_DEPLOYMENT, "input": texts}, retries=6)
        return [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]
    return _ollama("/api/embed", {"model": EMBED_MODEL, "input": texts, "truncate": True})["embeddings"]


def chat(system_prompt, user_prompt):
    """Send one question to the language model and return its answer."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if not USE_AZURE:
        data = _ollama("/api/chat", {"model": CHAT_MODEL, "stream": False, "messages": messages})
        return data["message"]["content"]

    payload = {"model": AZURE_CHAT_DEPLOYMENT, "messages": messages}
    if AZURE_REASONING_EFFORT:
        payload["reasoning_effort"] = AZURE_REASONING_EFFORT
    try:
        data = _azure("chat/completions", payload, retries=2)
    except LLMUnavailable:
        raise
    except LLMError as error:
        # Non-reasoning models reject this option; retry once without it.
        if "reasoning_effort" not in payload or "reasoning" not in str(error):
            raise
        del payload["reasoning_effort"]
        data = _azure("chat/completions", payload, retries=2)

    text = data["choices"][0]["message"].get("content")
    if not text:
        raise LLMError("The model returned an empty answer.")
    return text
