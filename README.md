# 📚 Multi-Modal RAG Pipeline

A production-style **Retrieval Augmented Generation (RAG)** system that lets users upload documents and ask questions. The system retrieves relevant chunks and generates grounded answers with an LLM.

**Stack:** FastAPI · LangChain · FAISS · Streamlit · Docker · Groq · Cohere

---

## Features

- **Document ingestion** — PDF upload, text extraction, smart chunking, metadata
- **Advanced retrieval** — Multi-query expansion, hybrid search (FAISS + BM25), Cohere reranking
- **Grounded answers** — Context-only prompts with source citations
- **Dockerized** — Backend + Streamlit UI, ready for cloud deploy

---

## Architecture

```
User → Streamlit UI → FastAPI → Ingestion → FAISS Vector Store
                              → Retrieval (MultiQuery + Hybrid + Rerank) → Groq LLM → Answer + Sources
```

---

## Project Structure

```
production-rag-system/
├── app/
│   ├── api/endpoints.py
│   ├── ingestion/pdf_ingestion_pipeline.py
│   ├── retrieval/          # retriever, vector_store, multiquery
│   ├── llm/llm_client.py
│   └── main.py
├── streamlit_app/app.py
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.streamlit
├── docker-compose.yml          # local dev
├── docker-compose.prod.yml     # VPS / single-host prod
├── render.yaml                 # Render.com one-click deploy
├── requirements.txt
├── requirements-streamlit.txt  # lightweight UI image
└── .env.example
```

---

## Setup (Local)

### 1. Clone and install

```bash
git clone https://github.com/Akash2512git/production-rag-system.git
cd production-rag-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | [Groq](https://console.groq.com/) API key |
| `COHERE_API_KEY` | Yes | [Cohere](https://dashboard.cohere.com/) API key |
| `API_URL` | Streamlit only | Backend URL (default: `http://localhost:8000/api`) |

### 3. Run locally

**Terminal 1 — backend:**
```bash
uvicorn app.main:app --reload
```

**Terminal 2 — UI:**
```bash
streamlit run streamlit_app/app.py
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  
- UI: http://localhost:8501  

---

## Docker

**Local dev** (with hot-reload volumes):
```bash
docker compose up --build
```

**Production** (no volume mounts):
```bash
docker compose -f docker-compose.prod.yml up --build -d
```

---

## Free-Tier Deployment (Recommended)

### Option 1: Render.com (easiest — use `render.yaml`)

Best for a **live demo link on your CV**. Two free web services: backend + Streamlit.

1. Push repo to GitHub
2. Sign up at [render.com](https://render.com)
3. **New → Blueprint** → connect repo → Render reads `render.yaml`
4. Set secrets when prompted:
   - `GROQ_API_KEY`
   - `COHERE_API_KEY`
5. Deploy. URLs will look like:
   - Backend: `https://rag-backend.onrender.com`
   - UI: `https://rag-streamlit.onrender.com`

**Free tier caveats:**
- Services **sleep after ~15 min** idle (first request wakes them — ~1–3 min cold start)
- Backend image is heavy (ML deps); if build/OOM fails, upgrade backend to **Starter ($7/mo)**
- Groq + Cohere free API tiers have rate limits

---

### Option 2: Hugging Face Spaces (UI) + Render (API)

Split deploy to keep the UI on HF's generous free tier.

| Component | Platform | Dockerfile |
|-----------|----------|------------|
| Backend | Render / Railway | `docker/Dockerfile.backend` |
| Streamlit UI | [Hugging Face Spaces](https://huggingface.co/spaces) | `docker/Dockerfile.streamlit` |

On HF Space, set **Settings → Variables:**
```
API_URL=https://your-backend.onrender.com
```

---

### Option 3: Fly.io

```bash
fly launch --dockerfile docker/Dockerfile.backend
fly secrets set GROQ_API_KEY=... COHERE_API_KEY=...
```

Deploy Streamlit as a second app with `API_URL` pointing to the backend. Fly gives ~$5/mo free credits (enough for a small demo).

---

### Option 4: Single VPS (Hetzner / DigitalOcean ~$4–6/mo)

Not free, but cheapest "real" prod setup:

```bash
# on VPS
git clone <repo> && cd production-rag-system
cp .env.example .env   # fill in keys
docker compose -f docker-compose.prod.yml up --build -d
```

Put **Caddy** or **nginx** in front for HTTPS.

---

## Deployment checklist

- [ ] `GROQ_API_KEY` and `COHERE_API_KEY` set on backend service
- [ ] `API_URL` on Streamlit points to backend (auto via `render.yaml` on Render)
- [ ] `/health` returns `{"status":"ok"}` on backend
- [ ] README live demo links updated
- [ ] Add demo URL to CV / LinkedIn

---

## API

### `POST /api/ingest`
Upload a PDF file (`multipart/form-data`, field: `file`).

### `POST /api/query`
```json
{ "query": "What is the transformer architecture?", "top_k": 5 }
```

### `GET /health`
Health check for load balancers and Render.

---

## Retrieval Pipeline

| Step | Technique |
|------|-----------|
| 1 | Multi-query expansion (Groq) |
| 2 | Hybrid retrieval — FAISS (70%) + BM25 (30%) |
| 3 | Cohere reranker (`rerank-english-v3.0`) |
| 4 | Top-k chunks → grounded LLM answer |

---

## Known limitations (v1)

- Vector index is **in-memory** — data is lost on restart
- No auth / rate limiting (fine for portfolio demo)
- First PDF ingest can be slow on free tier (model load + parsing)

---

## Author

**Akash Shukla** — AI / Backend Developer
