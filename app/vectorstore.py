from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "courseware_collection"


def _build_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Initialize and return a singleton Chroma vector store client.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = _build_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
