# Barta AI Service

Agentic RAG (Retrieval-Augmented Generation) backend for an AI-powered Bangladesh news chat application.

## Architecture

```
React Frontend
     │
     ▼ (HTTP + JWT)
┌─────────────────────────────────────────┐
│            FastAPI Backend               │
│  ┌──────────┐ ┌───────────┐ ┌────────┐ │
│  │  Auth     │ │   Chat    │ │ Health │ │
│  │Controller │ │Controller │ │Contrlr │ │
│  └──────────┘ └─────┬─────┘ └────────┘ │
│                      │        ┌────────┐│
│                      │        │ClickLog││
│                      │        │Contrlr ││
│                      │        └────────┘│
│         ┌────────────▼────────────┐     │
│         │  LangChain Agent        │     │
│         │  (Pluggable LLM:       │     │
│         │   OpenAI / Anthropic /  │     │
│         │   Google / Groq / Ollama)│     │
│         └────────────┬────────────┘     │
│                      │                   │
│    ┌─────────────────┼────────────────┐ │
│    │                 │                │ │
│  ┌─▼──────┐  ┌──────▼─────┐  ┌──────▼┐│
│  │Hybrid  │  │  Filter    │  │Latest ││
│  │Search  │  │  by Date/  │  │ News  ││
│  │(BM25 + │  │  Category  │  │       ││
│  │Vector) │  └──────┬─────┘  └───┬──┘│
│  └────┬────┘        │            │    │
│       │  ┌──────────┘────────────┘    │
│       │  │   Reciprocal Rank Fusion   │
│       └──┼────────────┐               │
│          ▼            ▼               │
│    ┌──────────────────────────────┐    │
│    │  MongoDB Atlas               │    │
│    │  • Vector Search (HNSW)     │    │
│    │  • Text Search (BM25)       │    │
│    │  • Click Logs (training)    │    │
│    │  (Pluggable Embeddings:     │    │
│    │   OpenAI / Cohere / Local)  │    │
│    └──────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Features

- **Hybrid Search**: Combines vector (semantic) and BM25 (keyword) retrieval with Reciprocal Rank Fusion (RRF) for best-of-both-worlds results
- **Agentic RAG**: LangChain agent decides which retrieval tools to use based on the query
- **Pluggable LLM**: Switch between OpenAI, Anthropic, Google Gemini, or Groq via `.env`
- **Pluggable Embeddings**: Choose OpenAI, Cohere (excellent Bangla), or free local models via `.env`
- **Semantic Search**: MongoDB Atlas Vector Search with configurable embedding models
- **BM25 Text Search**: MongoDB Atlas Search for keyword matching (exact names, entities, phrases)
- **Multi-tool Retrieval**: Search by topic, category, date range, tags, newspaper
- **Click-Log Tracking**: Logs query-article click pairs for future self-supervised embedding model training
- **JWT Authentication**: Secure token-based API access
- **Safety Guardrails**: Grounded responses, no fabrication, PII protection, content neutrality
- **Bilingual**: Supports both Bangla and English queries
- **Source Citations**: Every response includes referenced article sources
- **Controller Pattern**: Class-based controllers for clean API organization
- **Centralized Constants**: All strings in `constants.py` for easy localization

## Quick Start

### Prerequisites

- Python 3.12+
- MongoDB (local or Atlas)
- At least one LLM API key (OpenAI, Anthropic, Google, or Groq)

### 1. Setup

```bash
# Clone and enter directory
cd BartaAiService

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your MongoDB URI, API keys, and JWT secret
```

### 2. Choose Your LLM & Embedding Providers

Edit `.env` to select providers — only provide the API key for the one you use:

```env
# LLM: "openai", "anthropic", "google", "groq", or "ollama"
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# Embeddings: "openai", "cohere", or "local"
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

| LLM Provider | Example Models | Monthly Cost (1K req/day) |
|-------------|----------------|--------------------------|
| `openai` | gpt-4o, gpt-4o-mini, gpt-4.1-nano | $0.50–$18 |
| `anthropic` | claude-sonnet-4-20250514, claude-3-5-haiku | $4–$25 |
| `google` | gemini-2.0-flash, gemini-2.5-pro | $0.50–$16 |
| `groq` | llama-3.3-70b-versatile | $1–$2 |
| `ollama` | llama3, mistral, gemma2, phi3 | Free (local) |

| Embedding Provider | Example Models | Bangla Quality | Cost |
|-------------------|----------------|---------------|------|
| `openai` | text-embedding-3-small (1536d) | Good | API paid |
| `cohere` | embed-multilingual-v3.0 (1024d) | Excellent | API paid |
| `local` | intfloat/multilingual-e5-large (1024d) | Good | Free |

### 3. Free Local Development (Ollama + Local Embeddings)

You can run the entire stack **for free** with no API keys by using [Ollama](https://ollama.com/) for the LLM and local sentence-transformers for embeddings.

#### Install & Start Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.com/download for macOS / Linux / Windows
```

Start the Ollama server:

```bash
ollama serve
```

Pull a model (in a separate terminal):

```bash
# Recommended: lightweight and capable
ollama pull llama3

# Other good options:
ollama pull mistral
ollama pull gemma2
ollama pull phi3
```

#### Configure `.env` for Free Local Setup

Set these values in your `.env` file:

```env
# LLM — use Ollama (free, runs locally)
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434

# Embeddings — use local sentence-transformers (free, no API key)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# MongoDB (local via docker-compose)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=bartaAi

# JWT (generate your own secret)
JWT_SECRET_KEY=your-secret-key-here
```

| Local Embedding Model | Dimensions | Bangla Quality | Size |
|-----------------------|------------|---------------|------|
| `intfloat/multilingual-e5-large` | 1024 | Good | ~2.2 GB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fair | ~470 MB |

> **Note:** The first time you start the server with `EMBEDDING_PROVIDER=local`, the embedding model will be downloaded automatically. This may take a few minutes depending on model size and internet speed.

#### Run Everything Locally

```bash
# Terminal 1 — Start Ollama
ollama serve

# Terminal 2 — Start MongoDB (via Docker)
docker-compose up mongodb

# Terminal 3 — Start the API server
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The complete stack now runs locally with **zero API costs**.

### 4. Index Existing Articles

If you already have news articles in MongoDB without embeddings:

```bash
python -m scripts.index_articles
```

### 5. Create MongoDB Atlas Vector Search Index

In MongoDB Atlas, create a vector search index on the `news_articles` collection.
Set `numDimensions` to match your embedding model (1536 for OpenAI, 1024 for Cohere/E5, 384 for MiniLM):

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "Category"
    },
    {
      "type": "filter",
      "path": "NewsPaper"
    },
    {
      "type": "filter",
      "path": "PublishDate"
    }
  ]
}
```

Name it `news_vector_index` (or update `VECTOR_SEARCH_INDEX_NAME` in `.env`).

### 6. Create MongoDB Atlas Text Search Index

Create a separate Atlas Search index (not Vector Search) on the `news_articles` collection
for BM25 keyword matching. This is used by hybrid search alongside the vector index.

```json
{
  "name": "news_text_index",
  "mappings": {
    "dynamic": false,
    "fields": {
      "Title": { "type": "string", "analyzer": "lucene.standard" },
      "Body": { "type": "string", "analyzer": "lucene.standard" },
      "Tags": { "type": "string", "analyzer": "lucene.standard" }
    }
  }
}
```

Name it `news_text_index` (or update `TEXT_SEARCH_INDEX_NAME` in `.env`).

### 7. Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 8. Docker (Alternative)

```bash
docker-compose up --build
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | Service info |
| `GET` | `/api/v1/health` | No | Health check |
| `POST` | `/api/v1/auth/register` | No | Register a new user |
| `POST` | `/api/v1/auth/login` | No | Login and get JWT token |
| `POST` | `/api/v1/chat` | Yes | Send a message to the AI agent |
| `POST` | `/api/v1/chat/click-log` | Yes | Log query-article click pair |

### Chat Request Example

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the latest political news in Bangladesh?",
    "conversation_history": [],
    "session_id": null
  }'
```

### Chat Response Example

```json
{
  "answer": "Based on recent articles, here are the key political developments...",
  "sources": [
    {
      "title": "Parliament Session Highlights",
      "url": "https://example.com/article-1",
      "published": "2026-03-20",
      "newspaper": "daily_star"
    }
  ],
  "tool_calls": [
    {
      "tool": "semantic_news_search",
      "input": {"query": "latest political news Bangladesh"}
    }
  ],
  "session_id": "abc-123"
}
```

### Click-Log Request Example

When a user clicks on an article from the search results:

```bash
curl -X POST http://localhost:8000/api/v1/chat/click-log \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest political news Bangladesh",
    "news_id": "unique-article-id"
  }'
```

Or when the user clicks a source URL:

```bash
curl -X POST http://localhost:8000/api/v1/chat/click-log \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest political news Bangladesh",
    "source_url": "https://example.com/article-1"
  }'
```

### Click-Log Response Example

When clicked by `news_id` (returns the full article):

```json
{
  "message": "Click logged successfully.",
  "article": {
    "NewsId": "unique-article-id",
    "NewsPaper": "daily_star",
    "Category": "politics",
    "Title": "Parliament Session Highlights",
    "Body": "Full article text...",
    "Tags": ["politics", "parliament"],
    "PublishDate": "2026-03-20",
    "Author": "Reporter Name",
    "SourceURL": "https://example.com/article-1"
  }
}
```

When clicked by `source_url` (logs only, no article returned):

```json
{
  "message": "Click logged successfully.",
  "article": null
}
```

## Project Structure

```
BartaAiService/
├── app/
│   ├── main.py                  # FastAPI app entry point, middleware, lifecycle
│   ├── config.py                # Settings from environment variables
│   ├── constants.py             # Centralized strings (errors, prompts, metadata)
│   ├── auth/
│   │   ├── token.py             # JWT creation and verification
│   │   └── middleware.py        # Auth dependency (get_current_user)
│   ├── controllers/
│   │   ├── root_controller.py   # GET / — service info
│   │   ├── health_controller.py # GET /health — health check
│   │   ├── auth_controller.py   # POST /auth/register, /auth/login
│   │   └── chat_controller.py   # POST /chat, /chat/click-log
│   ├── models/
│   │   ├── news.py              # News article schemas
│   │   ├── chat.py              # Chat & click-log request/response schemas
│   │   └── user.py              # User & auth schemas
│   ├── services/
│   │   ├── embedding_service.py # Pluggable embeddings (OpenAI/Cohere/Local)
│   │   ├── llm_service.py       # Pluggable LLM (OpenAI/Anthropic/Google/Groq/Ollama)
│   │   └── news_service.py      # News article CRUD, click-log persistence
│   ├── agents/
│   │   ├── news_agent.py        # LangChain agentic RAG pipeline
│   │   └── tools.py             # Agent tools (search, filter, etc.)
│   └── database/
│       ├── mongodb.py           # MongoDB async connection management
│       └── vector_store.py      # Vector search, text search, hybrid search & RRF
├── scripts/
│   └── index_articles.py        # Batch embed existing articles
├── tests/
│   ├── integrationTest/         # Integration tests (API endpoint tests)
│   │   ├── conftest.py          # Test fixtures (async client)
│   │   ├── test_health.py       # Health & root endpoint tests
│   │   ├── test_auth.py         # Auth endpoint tests
│   │   └── test_chat.py         # Chat endpoint tests
│   └── unitTest/                # Unit tests (isolated function tests)
│       ├── agents/
│       │   ├── test_news_agent.py
│       │   └── test_tools.py
│       ├── auth/
│       │   ├── test_token.py
│       │   └── test_middleware.py
│       ├── controllers/
│       │   ├── test_auth_controller.py
│       │   ├── test_chat_controller.py
│       │   ├── test_health_controller.py
│       │   └── test_root_controller.py
│       ├── database/
│       │   ├── test_mongodb.py
│       │   └── test_vector_store.py
│       ├── models/
│       │   ├── test_chat.py
│       │   ├── test_news.py
│       │   └── test_user.py
│       ├── services/
│       │   ├── test_embedding_service.py
│       │   ├── test_llm_service.py
│       │   └── test_news_service.py
│       ├── test_config.py
│       └── test_constants.py
├── .coveragerc                  # Coverage configuration
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image
├── docker-compose.yml           # Local dev with MongoDB
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests
pytest tests/unitTest/ -v

# Run only integration tests
pytest tests/integrationTest/ -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=app --cov-report=html
# Open htmlcov/index.html in a browser to view
```

## React Frontend Integration

From your React app, connect to this backend:

```typescript
// api.ts
const API_BASE = 'http://localhost:8000/api/v1';

export async function login(email: string, password: string) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  return res.json(); // { access_token, token_type, expires_in }
}

export async function chat(token: string, message: string, history: Array<{role: string, content: string}> = []) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ message, conversation_history: history }),
  });
  return res.json(); // { answer, sources, tool_calls, session_id }
}

export async function logClick(token: string, query: string, newsId?: string, sourceUrl?: string) {
  const body: Record<string, string> = { query };
  if (newsId) body.news_id = newsId;
  if (sourceUrl) body.source_url = sourceUrl;

  const res = await fetch(`${API_BASE}/chat/click-log`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify(body),
  });
  return res.json(); // { message, article? }
}
```

## MongoDB Document Schema

Each news article in the `news_articles` collection:

```json
{
  "NewsId": "unique-id",
  "NewsPaper": "daily_star",
  "Category": "politics",
  "Title": "Article title",
  "Body": "Full article text...",
  "Tags": ["politics", "parliament"],
  "PublishDate": "2026-03-20T10:00:00",
  "Author": "Reporter Name",
  "SourceURL": "https://example.com/article",
  "embedding": [0.012, -0.034, ...]  // auto-generated, dimensions depend on model
}
```

Each click log in the `click_logs` collection (used for self-supervised training data):

```json
{
  "query": "latest political news Bangladesh",
  "news_id": "unique-article-id",
  "source_url": "https://example.com/article",
  "clicked_at": "2026-03-20T10:30:00+00:00"
}
```

## Security

- JWT tokens with expiration and signing
- Password hashing with bcrypt
- CORS restricted to configured origins (settings-driven)
- Rate limiting on API endpoints (settings-driven, toggleable)
- Non-root Docker user
- No sensitive data in error responses
- Input validation with Pydantic
- LLM guardrails: grounded responses, no fabrication, PII awareness
- Centralized error messages for consistent security posture
