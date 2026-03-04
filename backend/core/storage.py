# backend/core/storage.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — Storage Layer
# Owns all Pinecone + Redis connection logic.
# All dependencies injected from main.py via config.
# ─────────────────────────────────────────────────────────────

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.storage import RedisStore
from langchain_community.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_retriever(
    openai_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str,
    redis_url: str = "redis://localhost:6379",
    embedding_model: str = "text-embedding-3-small",
) -> ParentDocumentRetriever:
    """
    Builds and returns a ParentDocumentRetriever.

    Flow:
        Pinecone  - stores small 300-token child chunks (for precise search)
        Redis     - stores large 1500-token parent chunks (for LLM context)
        Retriever - searches child chunks, returns parent chunks to LLM
    
    Args:
        openai_api_key:      OpenAI API key for embedding model
        pinecone_api_key:    Pinecone API key for vector store
        pinecone_index_name: Name of the Pinecone index
        redis_url:           Redis connection string
        embedding_model:     OpenAI embedding model name

    Returns:
        ParentDocumentRetriever: Ready to use retriever instance
    """

    # 1. Embedding model
    #    Converts text -> 1536-dimensional dense vectors
    #    text-embedding-3-small: cheap, fast, great for RAG
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=openai_api_key # type: ignore
    )

    # 2. Pinecone vector store
    #    Holds the small 300-token child chunk embeddings
    #    dotproduct metric required for hybrid search (BM25 + Vector)
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )

    # 3. Redis document store
    #    Holds the full 1500-token parent chunks
    #    Pinecone returns doc_id -> Redis returns the full parent text
    redis_store = RedisStore(redis_url=redis_url)

    # 4. Parent splitter
    #    Large chunks — rich context fed to the LLM when answering
    #    overlap=150 prevents cutting sentences at chunk boundaries
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    # 5. Child splitter
    #    Small chunks — precise snippets embedded into Pinecone
    #    overlap=30 keeps enough context for accurate search
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    # 6. Wire everything into ParentDocumentRetriever
    #    id_key="doc_id" — metadata key linking child vector -> parent doc
    #    LangChain handles the Pinecone -> Redis lookup automatically
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        byte_store=redis_store,
        id_key="doc_id",
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )