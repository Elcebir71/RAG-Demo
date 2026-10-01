"""Central settings for the RAG demo. Change models or sizes here."""
import os

OLLAMA_URL = os.getenv("OLLAMA_URL","http://localhost:11434")

EMBED_MODEL = "bge-m3"     # multilingual embedding model (Dutch, Turkish, English, ...)
CHAT_MODEL = "llama3.2"    # small local LLM (about 2 GB)

DOCS_DIR = "docs"          # put your PDF / TXT / MD files here
DB_DIR = "db"              # the vector database is stored here
COLLECTION = "documents"

CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 150        # characters shared between neighbouring chunks
TOP_K = 4                  # number of chunks retrieved per question
