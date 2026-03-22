# backend/core/llm.py
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from backend.config import config

def get_llm():
    """Returns the LLM based on config provider."""
    if config.llm_provider == "mistral":
        return ChatMistralAI(api_key=config.mistral_api_key) # type: ignore
    
    # Latest openai model gpt-5.4-mini (released on March 2026)
    return ChatOpenAI(model="gpt-5.4-mini", api_key=config.openai_api_key) # type: ignore