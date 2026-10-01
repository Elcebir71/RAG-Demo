# rag-demo — Local RAG assistant for your own documents

A small, fully local **Retrieval-Augmented Generation (RAG)** pipeline in plain Python.
Drop PDFs into a folder, index them, and ask questions. Answers are grounded in your
documents and cite the exact file and page they came from.

- **No API keys, no cost, no data leaves your machine** — models run locally via [Ollama](https://ollama.com)
- **Multilingual** — ask in English, Dutch or Turkish; documents can be in any of them
- **Small and readable** — about 200 lines, no frameworks, every RAG step visible

## How it works

```
docs/  (PDF, TXT, MD)
   │  ingest.py   split into overlapping chunks → embed with bge-m3 → store in ChromaDB
   ▼
db/    (vector database)
   │  ask.py      embed the question → retrieve top-4 similar chunks → send to llama3.2
   ▼
Answer with source citations  [1] az-900.pdf, page 14
```

| File | Role |
|---|---|
| `config.py` | All settings: models, chunk size, overlap, number of retrieved chunks |
| `ollama_client.py` | Thin wrapper around the Ollama REST API (`/api/embed`, `/api/chat`) |
| `ingest.py` | **Indexing** — read documents, chunk, embed, store |
| `ask.py` | **Retrieval + generation** — find relevant chunks, build the prompt, answer |

## Example

```
> What is the difference between IaaS, PaaS and SaaS?

IaaS gives you virtual machines, storage and networking that you manage yourself,
PaaS adds a managed platform so you only deploy your code, and with SaaS you simply
use finished software ... [1][2]

Sources:
  [1] az-900.pdf, page 14  (similarity 0.71)
  [2] az-900.pdf, page 15  (similarity 0.66)
```

## Setup

**1. Install Ollama** from https://ollama.com/download and pull the models (once, about 3.2 GB):

```bash
ollama pull bge-m3
ollama pull llama3.2
```

**2. Create a virtual environment** (Python 3.10–3.12 recommended):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

**3. Add documents** to `docs/` and build the index:

```bash
python ingest.py
```

**4. Ask questions:**

```bash
python ask.py                                        # interactive mode
python ask.py "Hoeveel vakantiedagen krijg ik?"      # single question
```

Re-run `ingest.py` whenever you add or remove documents; it rebuilds the index from scratch.

## Design choices

- **Chunking** — 800 characters with 150 characters of overlap, cut at whitespace,
  so sentences on a chunk boundary are not lost.
- **Embeddings** — `bge-m3`, a multilingual model, so a Dutch question can match an English document.
- **Vector store** — ChromaDB with cosine distance, persisted to disk.
- **Grounding** — the system prompt tells the model to answer only from the retrieved
  context, to say "I don't know" otherwise, and to cite sources by number.
- **Robust indexing** — embeddings are sent in small batches; if a batch fails, chunks
  are retried one by one and only the failing ones are skipped, with Ollama's own
  error message shown.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot reach Ollama` | Start the Ollama app |
| `Model not found` | `ollama pull <model>` |
| `llama-server binary not found` | Ollama install is incomplete (often antivirus quarantine) — check Windows Security protection history, or reinstall Ollama |
| Chunks skipped with `NaN` | Switch `EMBED_MODEL` to `nomic-embed-text` in `config.py` and re-index |
| Slow answers | Normal on CPU; try `CHAT_MODEL = "llama3.2:1b"` |

## Roadmap

- [ ] Support `.docx` and scanned PDFs (OCR)
- [ ] REST API with FastAPI
- [ ] Docker image
- [ ] Cloud deployment on Azure (Container Apps, Azure AI Search, Azure OpenAI)

## Author

Hakan Şahin — [hakansahin.dev](https://hakansahin.dev)

Documents: Microsoft Learn (https://learn.microsoft.com), licensed under CC BY 4.0