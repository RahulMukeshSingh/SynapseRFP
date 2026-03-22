# backend/core/storage.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — Storage Layer
# Owns all Pinecone + Redis connection logic.
# All dependencies injected from main.py via config.
# ─────────────────────────────────────────────────────────────

import json
import uuid
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.storage import RedisStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.retrievers import PineconeHybridSearchRetriever

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.stores import BaseStore

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder


class HybridParentDocumentRetriever(BaseRetriever):
    """
    Custom retriever combining Pinecone Hybrid Search (BM25 + Dense)
    with the Parent-Child document pattern via Redis.
    """
    hybrid_retriever: PineconeHybridSearchRetriever
    byte_store: BaseStore[str, bytes]
    parent_splitter: TextSplitter
    child_splitter: TextSplitter
    id_key: str = "doc_id"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Search Pinecone for child chunks (BM25 + Dense vector match)
        child_docs = self.hybrid_retriever.invoke(query)
        
        # 2. Extract unique parent document IDs
        parent_ids = []
        for doc in child_docs:
            if self.id_key in doc.metadata and doc.metadata[self.id_key] not in parent_ids:
                parent_ids.append(doc.metadata[self.id_key])
                
        if not parent_ids:
            return []
            
        # 3. Retrieve full parent documents from Redis
        parent_docs_bytes = self.byte_store.mget(parent_ids)
        
        # 4. Reconstruct Document objects
        docs = []
        for doc_byte in parent_docs_bytes:
            if doc_byte:
                doc_dict = json.loads(doc_byte.decode("utf-8"))
                docs.append(
                    Document(
                        page_content=doc_dict["page_content"], 
                        metadata=doc_dict.get("metadata", {})
                    )
                )
                
        return docs

    def add_documents(self, documents: List[Document]) -> None:
        """Splits, encodes, and indexes documents into Redis and Pinecone."""
        parent_docs = self.parent_splitter.split_documents(documents)
        doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
        
        child_texts = []
        child_metadatas = []
        parent_key_values = []
        
        for doc_id, doc in zip(doc_ids, parent_docs):
            # Prepare parent doc for Redis storage
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            parent_key_values.append((doc_id, json.dumps(doc_dict).encode("utf-8")))
            
            # Split into children for Pinecone storage
            sub_docs = self.child_splitter.split_documents([doc])
            for sub_doc in sub_docs:
                child_texts.append(sub_doc.page_content)
                meta = sub_doc.metadata.copy()
                meta[self.id_key] = doc_id
                child_metadatas.append(meta)
                
       # Store child chunks in Pinecone in safe batches of 100
        batch_size = 100
        for i in range(0, len(child_texts), batch_size):
            batch_texts = child_texts[i : i + batch_size]
            batch_metas = child_metadatas[i : i + batch_size]
            
            # Upload batch to Pinecone
            self.hybrid_retriever.add_texts(
                texts=batch_texts, 
                metadatas=batch_metas
            )
        
        # Store parent documents in Redis in safe batches of 100
        for i in range(0, len(parent_key_values), batch_size):
            batch_parents = parent_key_values[i : i + batch_size]
            self.byte_store.mset(batch_parents)


def get_retriever(
    openai_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str,
    redis_url: str = "redis://localhost:6379",
    embedding_model: str = "text-embedding-3-small",
) -> HybridParentDocumentRetriever:
    """
    Builds and returns a HybridParentDocumentRetriever.
    """

    # 1. Dense Embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key) # type: ignore

    # 2. Sparse Embeddings (BM25)
    # Note: Using default() works, but fitting it to your specific corpus is highly recommended
    bm25_encoder = BM25Encoder().default()

    # 3. Pinecone Index setup
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    # 4. Initialize LangChain's dedicated Pinecone Hybrid Retriever
    hybrid_retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index
    )

    # 5. Redis document store
    redis_store = RedisStore(redis_url=redis_url)

    # 6. Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    # 7. Wire everything into our Custom Retriever
    return HybridParentDocumentRetriever(
        hybrid_retriever=hybrid_retriever,
        byte_store=redis_store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        id_key="doc_id"
    )