from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================
    # LLM
    # ==========================

    llm_provider: str = Field(default="cerebras")
    cerebras_api_key: str = Field(...)
    llm_model: str = Field(default="qwen-3-32b")

    # ==========================
    # Embeddings
    # ==========================

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # ==========================
    # Reranker
    # ==========================

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ==========================
    # Retrieval
    # ==========================

    top_k: int = Field(default=10)
    final_top_k: int = Field(default=5)

    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ==========================
    # Paths
    # ==========================

    data_dir: str = Field(default="data")
    raw_data_dir: str = Field(default="data/raw")
    processed_data_dir: str = Field(default="data/processed")
    index_dir: str = Field(default="data/indexes")

    # ==========================
    # Database
    # ==========================

    database_url: str = Field(default="sqlite:///medsearch.db")

    # ==========================
    # Logging
    # ==========================

    log_level: str = Field(default="INFO")

    # ==========================
    # MLflow
    # ==========================

    mlflow_tracking_uri: str = Field(default="mlruns")


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()


settings = get_settings()