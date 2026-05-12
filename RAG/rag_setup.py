# RAG/rag_setup.py
"""
Khởi tạo ChromaDB + SentenceTransformer.
Module-level globals để các file khác import trực tiếp (backward compat).
Python chỉ load module 1 lần → embedder/collection là singleton tự nhiên.
"""

from sentence_transformers import SentenceTransformer
import chromadb

# ── Embedder ───────────────────────────────────────────────────────────────────
embedder = SentenceTransformer("keepitreal/vietnamese-sbert")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
# Path tương đối so với CWD khi chạy (project root)
chroma_client = chromadb.PersistentClient(path="./medical_db")

collection = chroma_client.get_or_create_collection(
    name="medical_articles",
    metadata={"hnsw:space": "cosine"},
)


# ── Helper functions (dùng trong rag_server.py) ────────────────────────────────
def get_embedder() -> SentenceTransformer:
    """Trả về embedder đã load sẵn."""
    return embedder


def get_collection():
    """Trả về ChromaDB collection."""
    return collection