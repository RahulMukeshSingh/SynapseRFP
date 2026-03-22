# backend/ingestion/ingest.py
# ─────────────────────────────────────────────────────────────
# SynapseRFP — Offline Data Ingestion Pipeline
# Loads PDFs from the data/ directory and pushes them 
# through the ParentDocumentRetriever into Pinecone & Redis.
# ─────────────────────────────────────────────────────────────

import os
import sys
import logging
from pathlib import Path
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    Docx2txtLoader,       # .docx
    UnstructuredExcelLoader,  # .xlsx
    TextLoader           # .txt .md
)

# 1. Configure the Logger
# This formats logs as: "2026-03-05 14:30:00 [INFO] connecting to pinecone..."
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Append the root 'SYNAPSERFP' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.core import get_retriever
from backend.config import config

# maps file extension to the right loader
LOADERS = {
    ".pdf":  UnstructuredPDFLoader,
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
}

def load_documents(data_directory: str):
    """
    Loads all supported files from data/ directory.
    Skips anything we don't have a loader for.
    """
    documents = []
    data_path = Path(data_directory)

    # grab every file in the folder regardless of extension
    all_files = [f for f in data_path.iterdir() if f.is_file()]

    if not all_files:
        logger.warning(f"No files found in {data_directory}")
        return documents

    for file_path in all_files:
        ext = file_path.suffix.lower()

        # skip files we don't support yet
        if ext not in LOADERS:
            logger.warning(f"Skipping {file_path.name} — unsupported format ({ext})")
            continue

        try:
            logger.info(f"Loading: {file_path.name}")
            loader = LOADERS[ext](str(file_path))
            documents.extend(loader.load())
        except Exception as e:
            # Use exception level to automatically capture the stack trace if needed, 
            # or just error to keep it clean.
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    return documents


def ingest_data():
    """main ingestion pipeline — loads files, chunks, pushes to pinecone + redis"""

    logger.info("Connecting to Pinecone and Redis...")

    retriever = get_retriever(
        openai_api_key=config.openai_api_key,
        pinecone_api_key=config.pinecone_api_key,
        pinecone_index_name=config.pinecone_index_name,
        redis_url=config.redis_url,
        embedding_model=config.openai_embedding_model
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    logger.info("Scanning for documents...")
    raw_documents = load_documents(data_dir)

    if not raw_documents:
        logger.info("Nothing to ingest - drop some files into the data/ folder.")
        return

    logger.info(f"Chunking and embedding {len(raw_documents)} pages...")
    logger.info("This might take a minute depending on document size...")

    # splits into parent chunks -> redis
    # splits into child chunks -> embeds -> pinecone
    # links them via doc_id automatically
    retriever.add_documents(raw_documents)

    logger.info("Done — data is successfully loaded into Pinecone and Redis.")

if __name__ == "__main__":
    ingest_data()