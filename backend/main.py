# backend/main.py
from config import config
from core import get_retriever

retriever = get_retriever(
    openai_api_key=config.openai_api_key,
    pinecone_api_key=config.pinecone_api_key,
    pinecone_index_name=config.pinecone_index_name,
    redis_url=config.redis_url,
    embedding_model=config.openai_embedding_model,
)