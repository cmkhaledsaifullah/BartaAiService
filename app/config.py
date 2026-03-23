from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # MongoDB
    mongodb_uri: str
    mongodb_db_name: str = "bartaAi"
    mongodb_tls_cert_key_file: str = ""

    # OpenAI (required if llm_provider=openai or embedding_provider=openai)
    openai_api_key: str = ""

    # LLM Provider: "openai", "anthropic", "google", "groq", or "ollama"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"

    # Anthropic (required only if llm_provider=anthropic)
    anthropic_api_key: str = ""

    # Google (required only if llm_provider=google)
    google_api_key: str = ""

    # Groq (required only if llm_provider=groq)
    groq_api_key: str = ""

    # Ollama (required only if llm_provider=ollama)
    ollama_base_url: str = "http://localhost:11434"

    # Embedding Provider: "openai", "cohere", or "local"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # Cohere (required only if embedding_provider=cohere)
    cohere_api_key: str = ""

    # JWT
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # Vector Search
    vector_search_index_name: str = "news_vector_index"
    vector_search_num_candidates: int = 100
    vector_search_limit: int = 10

    # Text Search (Atlas Search / BM25)
    text_search_index_name: str = "news_text_index"

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "GET,POST,OPTIONS"
    cors_allow_headers: str = "Authorization,Content-Type"

    # Rate Limiting
    rate_limit: str = "20/minute"
    rate_limit_enabled: bool = True

    # App
    app_env: str = "development"
    app_debug: bool = False
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def cors_allow_methods_list(self) -> list[str]:
        return [m.strip() for m in self.cors_allow_methods.split(",")]

    @property
    def cors_allow_headers_list(self) -> list[str]:
        return [h.strip() for h in self.cors_allow_headers.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
