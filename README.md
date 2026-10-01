# rag-demo — Kendi belgelerinle konuşan yerel RAG asistanı

Tamamen bilgisayarında çalışır: API anahtarı yok, ücret yok, belgeler dışarı gitmez.

```
docs/ (PDF, TXT, MD)
   │  ingest.py   → parçalara böl → bge-m3 ile vektöre çevir → ChromaDB'ye kaydet
   ▼
db/  (vektör veritabanı)
   │  ask.py      → soruyu vektöre çevir → en benzer 4 parçayı bul → llama3.2'ye ver
   ▼
Kaynak gösterilmiş cevap
```

## Dosyalar

| Dosya | Görevi |
|---|---|
| `config.py` | Tüm ayarlar (model adları, parça boyutu, kaç parça getirileceği) |
| `ollama_client.py` | Ollama'ya iki çağrı: `embed` (vektör) ve `chat` (cevap) |
| `ingest.py` | Adım 1 – İndeksleme |
| `ask.py` | Adım 2 + 3 – Arama ve cevap üretme |

## Kurulum

**1. Ollama'yı kur** — https://ollama.com/download (Windows/macOS/Linux). Kurduktan sonra uygulama arka planda çalışır.

**2. Modelleri indir** (bir kere, toplam ~3,2 GB):

```bash
ollama pull bge-m3
ollama pull llama3.2
```

**3. Python ortamını hazırla** (Python 3.10+):

```bash
cd rag-demo
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

**4. Belgelerini `docs/` klasörüne koy** (PDF, .txt veya .md).

**5. İndeksle:**

```bash
python ingest.py
```

**6. Soru sor:**

```bash
python ask.py                                   # sohbet modu
python ask.py "Hoeveel vakantiedagen krijg ik?" # tek soru
```

Belge ekleyip çıkardığında `ingest.py`'yi tekrar çalıştır; veritabanını baştan kurar.

## Denemeler (öğrenmek için)

- `config.py` içinde `CHUNK_SIZE` değerini 300 ve 1500 yapıp aynı soruyu sor. Cevap nasıl değişiyor?
- `TOP_K` değerini 1 ve 8 yap. Az parça mı daha iyi, çok parça mı?
- Belgelerde olmayan bir şey sor. Model "bilmiyorum" diyor mu?
- `ask.py` içindeki `SYSTEM_PROMPT`'u değiştir ve davranışı gözlemle.
- Bilgisayar yavaşsa `CHAT_MODEL = "llama3.2:1b"`, daha güçlüyse `"qwen2.5:7b"` dene.

## Sonraki adımlar (bulut)

Aynı yapı bulutta şöyle karşılık bulur: ChromaDB → Azure AI Search / pgvector,
Ollama → Azure OpenAI / AWS Bedrock, `ask.py` → FastAPI ile bir web API + Docker + bulut deployment.
