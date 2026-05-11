# rag_setup.py
from sentence_transformers import SentenceTransformer
import chromadb

# Model embed tiếng Việt
embedder = SentenceTransformer("keepitreal/vietnamese-sbert")

# Khởi tạo ChromaDB lưu xuống disk
chroma_client = chromadb.PersistentClient(path="./medical_db")
collection = chroma_client.get_or_create_collection(
    name="medical_articles",
    metadata={"hnsw:space": "cosine"}
)