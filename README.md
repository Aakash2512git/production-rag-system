# 📚 Multi-Modal RAG Pipeline

A production-style **Retrieval Augmented Generation (RAG)** system that allows users to upload documents and ask questions about them. The system retrieves relevant document chunks from a vector database and generates grounded answers using an LLM.

This project demonstrates how to build an **end-to-end AI application** using FastAPI, LangChain, ChromaDB, and Streamlit.

---

# 🚀 Features

### 📄 Document Ingestion
- Upload **PDF and TXT documents**
- Automatic text extraction and chunking
- Metadata preservation for source attribution

### 🔎 Advanced Retrieval Pipeline
- Multi-Query Retrieval
- Hybrid Search (vector + lexical)
- Reciprocal Rank Fusion (RRF)
- Cohere reranking for better relevance

### 🤖 LLM Answer Generation
- Context-aware answers
- Reduced hallucination using grounded prompts
- Source citations returned with answers

### 💬 Interactive UI
Built with Streamlit

- Upload documents
- Ask questions
- View retrieved source chunks

### 🐳 Dockerized Deployment
The system runs with:

- FastAPI backend
- Streamlit frontend
- Docker + Docker Compose

---

# 🏗️ Architecture

```
User
 │
 ▼
Streamlit UI
 │
 ▼
FastAPI Backend
 │
 ▼
Document Ingestion Pipeline
 │
 ▼
Vector Database (ChromaDB)
 │
 ▼
Retrieval Pipeline
 ├─ MultiQuery Retriever
 ├─ Hybrid Search
 ├─ RRF Ranking
 └─ Cohere Reranker
 │
 ▼
LLM (Groq)
 │
 ▼
Answer + Sources
```

---

# 📂 Project Structure

```
rag_pipeline
│
├── app
│   ├── api
│   │   └── endpoints.py
│   │
│   ├── ingestion
│   │   └── pdf_ingestion_pipeline.py
│   │
│   ├── retrieval
│   │   ├── multiquery.py
│   │   └── retriever.py
│   │
│   ├── llm
│   │   └── llm_client.py
│   │
│   └── main.py
│
├── streamlit_app
│   └── app.py
│
├── docker
│   ├── Dockerfile.backend
│   └── Dockerfile.streamlit
│
├── docker-compose.yml
├── requirements.txt
├── notebooks
├── tests
└── README.md
```

---

# ⚙️ Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/rag_pipeline.git
cd rag_pipeline
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
```

---

# ▶️ Run Locally

Start the backend server:

```bash
uvicorn app.main:app --reload
```

Start Streamlit UI:

```bash
streamlit run streamlit_app/app.py
```

Backend API docs:

```
http://localhost:8000/docs
```

Streamlit UI:

```
http://localhost:8501
```

---

# 🐳 Run with Docker

Build and run the entire system:

```bash
docker compose up --build
```

Services will be available at:

Backend

```
http://localhost:8000
```

Streamlit

```
http://localhost:8501
```

---

# 📡 API Endpoints

### Ingest Documents

```
POST /api/ingest
```

Upload PDF or TXT files.

---

### Query Documents

```
POST /api/query
```

Example request:

```json
{
  "query": "What is the transformer architecture?"
}
```

Example response:

```json
{
  "answer": "Transformers are neural networks designed for sequence modeling...",
  "sources": [
    "Document chunk 1",
    "Document chunk 2"
  ]
}
```

---

# 🧠 Retrieval Strategy

The system improves retrieval accuracy using multiple techniques.

| Technique | Purpose |
|----------|--------|
| MultiQuery Retrieval | Generate multiple query variations |
| Hybrid Search | Combine vector similarity and keyword search |
| Reciprocal Rank Fusion | Merge rankings from different retrievers |
| Cohere Reranker | Improve final relevance ranking |

---

# 📊 Tech Stack

Backend
- FastAPI
- LangChain

Vector Database
- ChromaDB

LLM
- Groq API

Frontend
- Streamlit

Deployment
- Docker
- Docker Compose

---

# 📈 Future Improvements

- Conversational memory
- Streaming LLM responses
- Multi-modal document ingestion
- Evaluation metrics (RAGAS)
- Authentication and user sessions

---

# 👨‍💻 Author

**Akash Shukla**

AI / Backend Developer

---

# ⭐ Why This Project Matters

This project demonstrates:

- End-to-end LLM application design
- Retrieval engineering
- Production API development
- Containerized deployment
- AI system architecture

These skills are essential for modern **AI and backend engineering roles**.