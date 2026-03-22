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
│                      │                   │
│         ┌────────────▼────────────┐     │
│         │  LangChain Agent        │     │
│         │  (Pluggable LLM:       │     │
│         │   OpenAI / Anthropic /  │     │
│         │   Google / Groq)        │     │
│         └────────────┬────────────┘     │
│                      │                   │
│    ┌─────────────────┼────────────────┐ │
│    │                 │                │ │
│  ┌─▼──────┐  ┌──────▼─────┐  ┌──────▼┐│
│  │Semantic │  │  Filter    │  │Latest ││
│  │Search   │  │  by Date/  │  │ News  ││
│  │(Vector) │  │  Category  │  │       ││
│  └────┬────┘  └──────┬─────┘  └───┬──┘│
│       └──────────────┼─────────────┘   │
│                      ▼                  │
│    ┌──────────────────────────────┐    │
│    │  MongoDB + Vector Search     │    │
│    │  (Pluggable Embeddings:     │    │
│    │   OpenAI / Cohere / Local)  │    │
│    └──────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Features

- **Agentic RAG**: LangChain agent decides which retrieval tools to use based on the query
- **Pluggable LLM**: Switch between OpenAI, Anthropic, Google Gemini, or Groq via `.env`
- **Pluggable Embeddings**: Choose OpenAI, Cohere (excellent Bangla), or free local models via `.env`
- **Semantic Search**: MongoDB Atlas Vector Search with configurable embedding models
- **Multi-tool Retrieval**: Search by topic, category, date range, tags, newspaper
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
# LLM: "openai", "anthropic", "google", or "groq"
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

| Embedding Provider | Example Models | Bangla Quality | Cost |
|-------------------|----------------|---------------|------|
| `openai` | text-embedding-3-small (1536d) | Good | API paid |
| `cohere` | embed-multilingual-v3.0 (1024d) | Excellent | API paid |
| `local` | intfloat/multilingual-e5-large (1024d) | Good | Free |

### 3. Index Existing Articles

If you already have news articles in MongoDB without embeddings:

```bash
python -m scripts.index_articles
```

### 4. Create MongoDB Atlas Vector Search Index

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
      "path": "CategoryId"
    },
    {
      "type": "filter",
      "path": "NewsPaperId"
    },
    {
      "type": "filter",
      "path": "PublishDate"
    }
  ]
}
```

Name it `news_vector_index` (or update `VECTOR_SEARCH_INDEX_NAME` in `.env`).

### 5. Run the Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Docker (Alternative)

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
│   │   └── chat_controller.py   # POST /chat — AI chat (authenticated)
│   ├── models/
│   │   ├── news.py              # News article schemas
│   │   ├── chat.py              # Chat request/response schemas
│   │   └── user.py              # User & auth schemas
│   ├── services/
│   │   ├── embedding_service.py # Pluggable embeddings (OpenAI/Cohere/Local)
│   │   ├── llm_service.py       # Pluggable LLM (OpenAI/Anthropic/Google/Groq)
│   │   └── news_service.py      # News article CRUD queries
│   ├── agents/
│   │   ├── news_agent.py        # LangChain agentic RAG pipeline
│   │   └── tools.py             # Agent tools (search, filter, etc.)
│   └── database/
│       ├── mongodb.py           # MongoDB async connection management
│       └── vector_store.py      # Vector search operations
├── scripts/
│   └── index_articles.py        # Batch embed existing articles
├── tests/
│   ├── conftest.py              # Test fixtures
│   ├── test_health.py           # Health & root endpoint tests
│   ├── test_auth.py             # Auth endpoint tests
│   └── test_chat.py             # Chat endpoint tests
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container image
├── docker-compose.yml           # Local dev with MongoDB
└── README.md
```

## Testing

```bash
pytest tests/ -v
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
```

## MongoDB Document Schema

Each news article in the `news_articles` collection:

```json
{
  "NewsId": "unique-id",
  "NewsPaperId": "daily_star",
  "CategoryId": "politics",
  "Title": "Article title",
  "Body": "Full article text...",
  "Tags": ["politics", "parliament"],
  "PublishDate": "2026-03-20T10:00:00",
  "Author": "Reporter Name",
  "SourceURL": "https://example.com/article",
  "embedding": [0.012, -0.034, ...]  // auto-generated, dimensions depend on model
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
