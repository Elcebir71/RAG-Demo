- **Runs fully local by default** — models run via [Ollama](https://ollama.com); set Azure OpenAI variables to run in the cloud

A small, fully local **Retrieval-Augmented Generation (RAG)** pipeline in plain Python.
Drop PDFs into a folder, index them, and ask questions. Answers are grounded in your
documents and cite the exact file and page they came from.

- **No API keys, no cost, no data leaves your machine** — models run locally via [Ollama](https://ollama.com)
- **Multilingual** — ask in English, Dutch or Turkish; documents can be in any of them
- **Small and readable** — no frameworks, every RAG and agent step visible in plain Python

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
  | `llm_client.py` | Model backend: Ollama locally, Azure OpenAI in the cloud; chat, embeddings and tool calling |
| `ingest.py` | **Indexing** — read documents, chunk, embed, store |
| `ask.py` | **Retrieval + generation** — find relevant chunks, build the prompt, answer |
| `tools.py` | Tool registry: schemas, risk levels, validation and business rules |
| `store.py` | SQLite state: actions, audit log, notes, outbox |
| `agent.py` | Agent loop, approve and reject |
| `tests/` | Tests for the guarantees, with a scripted fake model |

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

## Agent with guarded actions

The assistant can also act: search the documents, save notes and send email.
The design follows one rule: **the model proposes, the application decides.**
A prompt guides the model's reasoning, but permissions, validation, approval and
recovery are enforced in code, where no prompt or document can change them.

```
request ─▶ model proposes tool calls
              │
              ▼
        tools.validate()  ── unknown tool, bad arguments,
              │              recipient not on allowlist ──▶ rejected (logged)
              ▼
         risk level?
     read / low ──▶ run now ──▶ done / failed (logged)
     high ───────▶ pending ──▶ human approves with admin key ──▶ re-check ──▶ run
```

| Tool | Risk | What the application does |
|---|---|---|
| `search_documents` | read | Runs automatically |
| `save_note` | low | Runs automatically and is logged |
| `send_email` | high | Waits for approval; recipient must be on `EMAIL_ALLOWLIST` |

**Guarantees, each covered by a test in `tests/`**

- Unknown tools, extra fields and malformed arguments are rejected (strict Pydantic schemas).
- A recipient outside the allowlist is rejected, even when a document contains
  an injected instruction and the model follows it.
- Nothing high-risk runs without approval, and approval needs `ADMIN_API_KEY`.
  Without a configured key, approvals are disabled (fail closed).
- Approving twice runs once: every status change is a conditional update.
- Rules are checked again at approval time, not only at proposal time.
- Failures are recorded, never lost; the loop stops after `MAX_STEPS` rounds.
- Every status change is written to an audit log.

Email runs in **demo mode**: an approved email goes to an `outbox` table and is not
sent. A public demo that sends real email could be abused as a spam relay.

**Model choice, measured, not assumed.** Same tool definition, three runs each:

| Model | System prompt | Unwanted tool calls | Correct tool calls |
|---|---|---|---|
| llama3.2 (3B) | no | 3/3 | 3/3 |
| llama3.2 (3B) | yes | 1/3 | 3/3 |
| qwen2.5:7b | no | 0/3 | 3/3 |
| qwen2.5:7b | yes | 0/3 | 3/3 |

`qwen2.5:7b` is the default (`AGENT_MODEL`). Known limitation: after a document
search it often asks the user for confirmation instead of proposing the next
action. The guarantees above do not depend on the model; its usefulness does.

**Try it**

```bash
# PowerShell: $env:EMAIL_ALLOWLIST = "you@example.com"; $env:ADMIN_API_KEY = "local-test-key"
python agent.py "Email you@example.com with subject 'Test' and body 'Hello'"
python agent.py --approve <id>   # the id printed by the previous command
python -m pytest -q          # needs: pip install -r requirements-dev.txt
```

API: `POST /agent`, and with the `x-admin-key` header: `GET /actions?status=pending`,
`POST /actions/{id}/approve`, `POST /actions/{id}/reject`, `GET /actions/{id}/audit`.

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