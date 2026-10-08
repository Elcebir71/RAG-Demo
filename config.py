"""Central settings for the RAG demo. Change models or sizes here."""

import os

# --- Backend selection -------------------------------------------------------
# If an Azure OpenAI endpoint is set in the environment, the app uses Azure.
# Otherwise it falls back to a local Ollama server. No secrets live in code.
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "chat")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")
AZURE_REASONING_EFFORT = os.getenv("AZURE_REASONING_EFFORT", "low")  # "" = do not send

USE_AZURE = bool(AZURE_ENDPOINT)
BACKEND = "azure" if USE_AZURE else "ollama"

# --- Local backend (Ollama) --------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "bge-m3"     # multilingual embedding model (Dutch, Turkish, English, ...)
CHAT_MODEL = "llama3.2"    # small local LLM (about 2 GB)

# --- Documents and index -----------------------------------------------------
DOCS_DIR = "docs"          # put your PDF / TXT / MD files here
DB_DIR = "db"              # the vector database is stored here

# Vectors from different embedding models are not comparable (and have different
# sizes), so each backend gets its own collection inside the same database.
COLLECTION = "documents-azure" if USE_AZURE else "documents"

CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 150        # characters shared between neighbouring chunks
TOP_K = 4                  # number of chunks retrieved per question
# --- Agent -------------------------------------------------------------------
AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3.2")  # local model used for tool calling
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen2.5:7b")  # chosen by a small tool-calling test, see README

# Email recipients the agent may ever propose. Comma separated. Empty = no email at all.
EMAIL_ALLOWLIST = {a.strip().lower() for a in os.getenv("EMAIL_ALLOWLIST", "").split(",") if a.strip()}
AGENT_DB = os.getenv("AGENT_DB", "agent.db")  # SQLite file for actions, audit log, notes and outbox
# Key for the approval endpoints. Empty = approvals are disabled (fail closed).
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")