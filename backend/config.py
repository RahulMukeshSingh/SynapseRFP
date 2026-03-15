# backend/config.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — Configuration
# Single place where .env is read.
# Every other file imports from here — never calls os.getenv()
# ─────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()


@dataclass
class Config:

    # OpenAI

    # field() delays os.getenv() until Config() is called
    # by then load_dotenv() has already run -> key exists
    # without it, Python reads env at import time -> None
    openai_api_key: str         = field(default_factory=lambda: os.getenv("OPENAI_API_KEY",""))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    # Mistral
    mistral_api_key: str = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))

    # Cohere
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))

    # Pinecone
    pinecone_api_key: str       = field(default_factory=lambda: os.getenv("PINECONE_API_KEY",""))
    pinecone_index_name: str    = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "rfp-responder"))

    # Redis
    redis_url: str              = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))

    # LLM
    llm_provider: str           = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))


# single instance — import this everywhere
config = Config()
