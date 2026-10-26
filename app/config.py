"""Application configuration module.

Uses Pydantic BaseSettings to load environment variables cleanly and safely.
No real secrets are stored or logged.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Core settings for HR Policy RAG Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core Application Settings
    app_name: str = Field(
        default="HR Policy RAG Assistant",
        description="Application display name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application release version",
    )
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Runtime environment identifier",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode for verbose diagnostics",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )
    port: int = Field(
        default=8501,
        description="Server port for Streamlit",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )

    # Document Ingestion Settings
    chunk_size: int = Field(
        default=600,
        description="Target maximum character length per chunk",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Character overlap between consecutive chunks",
    )
    raw_data_dir: str = Field(
        default="data/raw",
        description="Directory containing raw policy documents",
    )
    processed_data_dir: str = Field(
        default="data/processed",
        description="Directory for persisting processed chunk datasets",
    )

    # Vector Database (Qdrant) Settings
    qdrant_url: Optional[str] = Field(
        default=None,
        description="Qdrant cluster URL (e.g. http://localhost:6333, https://xyz.cloud.qdrant.io, or :memory:)",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API Key for cloud/authenticated instances",
    )
    qdrant_collection_name: str = Field(
        default="hr_policies",
        description="Qdrant collection name for HR policies",
    )

    # Embedding Settings
    embedding_provider: str = Field(
        default="fastembed",
        description="Embedding provider identifier ('fastembed' or 'mock')",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(
        default=384,
        description="Embedding vector dimensionality",
    )

    # LLM Settings
    llm_provider: str = Field(
        default="router",
        description="LLM provider: 'router' (LiteLLM Vertex -> Groq fallback), 'gemini', 'groq', 'openai', or 'mock'",
    )
    llm_model: str = Field(
        default="gemini-1.5-pro",
        description="Model identifier (e.g. gemini-1.5-pro, llama-3.3-70b-versatile, gpt-4o-mini)",
    )
    google_cloud_project: Optional[str] = Field(
        default=None,
        description="Google Cloud Project ID for Vertex AI Gemini",
    )
    google_cloud_location: str = Field(
        default="us-central1",
        description="Google Cloud Location for Vertex AI Gemini (e.g. us-central1)",
    )
    vertex_model: str = Field(
        default="gemini-1.5-pro",
        description="Vertex AI Gemini model identifier (e.g. gemini-1.5-pro, gemini-1.5-flash)",
    )
    groq_model: str = Field(
        default="openai/gpt-oss-20b",
        description="Groq fallback model identifier (e.g. openai/gpt-oss-20b, openai/gpt-oss-120b)",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API Key (for direct Google AI Studio API)",
    )
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API Key for fallback generation",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API Key",
    )
    llm_temperature: float = Field(
        default=0.2,
        description="LLM sampling temperature",
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum generation token budget",
    )

    # Hybrid Search & Reranking Settings
    jina_api_key: Optional[str] = Field(
        default=None,
        description="Jina AI API Key for cross-encoder reranking",
    )
    jina_reranker_model: str = Field(
        default="jina-reranker-v2-base-multilingual",
        description="Jina reranker model identifier",
    )
    hybrid_top_k: int = Field(
        default=20,
        description="Number of candidate chunks retrieved by hybrid fusion before reranking",
    )
    rerank_top_k: int = Field(
        default=5,
        description="Number of top chunks passed to LLM after reranking",
    )
    fusion_strategy: str = Field(
        default="rrf",
        description="Fusion strategy: 'rrf' (Reciprocal Rank Fusion) or 'weighted'",
    )
    rrf_k: int = Field(
        default=60,
        description="Smoothing constant k for Reciprocal Rank Fusion",
    )
    dense_weight: float = Field(
        default=0.6,
        description="Weight for dense retrieval in weighted fusion",
    )
    bm25_weight: float = Field(
        default=0.4,
        description="Weight for BM25 retrieval in weighted fusion",
    )

    # Semantic Cache & Memory Settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable semantic caching of policy query answers",
    )
    cache_similarity_threshold: float = Field(
        default=0.90,
        description="Cosine similarity threshold for semantic cache hits",
    )
    current_policy_version: str = Field(
        default="2026",
        description="Current active policy version to invalidate stale cache entries",
    )
    memory_max_turns: int = Field(
        default=5,
        description="Maximum conversational query-response turns retained in session memory",
    )

    # Security & Google Model Armor Settings
    model_armor_enabled: bool = Field(
        default=True,
        description="Enable Google Model Armor and local security screening",
    )
    model_armor_project: Optional[str] = Field(
        default=None,
        description="Google Cloud Project ID for Google Model Armor API",
    )
    model_armor_location: str = Field(
        default="us-central1",
        description="Google Cloud region for Model Armor API",
    )
    model_armor_template: Optional[str] = Field(
        default=None,
        description="Google Model Armor floor setting or template identifier",
    )
    security_strict_mode: bool = Field(
        default=True,
        description="Block suspected prompt injections and unsafe outputs strictly",
    )
    max_input_length: int = Field(
        default=2000,
        description="Maximum permissible query character length before truncation/blocking",
    )

    # Google OAuth / OIDC Authentication & Authorization Settings
    google_client_id: Optional[str] = Field(
        default=None,
        description="Google Cloud OAuth 2.0 Web Client ID",
    )
    google_client_secret: Optional[str] = Field(
        default=None,
        description="Google Cloud OAuth 2.0 Web Client Secret",
    )
    google_redirect_uri: str = Field(
        default="http://localhost:8501",
        description="Authorized redirect URI for Google OAuth callback",
    )
    allowed_email_domains: str = Field(
        default="company.com,example.com",
        description="Comma-separated corporate email domains authorized to access the assistant",
    )
    allowed_users: str = Field(
        default="",
        description="Comma-separated individual employee emails explicitly authorized to access the assistant",
    )
    admin_users: str = Field(
        default="admin@company.com",
        description="Comma-separated administrator emails authorized to perform policy ingestion and management",
    )

    # Production Document Storage (Google Cloud Storage) Settings
    gcs_bucket_name: str = Field(
        default="company-hr-policies",
        description="GCS bucket name for authoritative production policy storage",
    )
    gcs_project: Optional[str] = Field(
        default=None,
        description="Google Cloud project ID hosting the GCS bucket",
    )
    gcs_local_root: str = Field(
        default="data/storage/gcs_mock",
        description="Local directory backing GCS operations when cloud credentials are unconfigured",
    )

    # Observability & LangSmith Settings
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith execution tracing and evaluation tracking",
    )
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API Key for cloud tracing and evaluation datasets",
    )
    langsmith_project: str = Field(
        default="hr-policy-rag",
        description="LangSmith project name for traces and runs",
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith endpoint URL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
