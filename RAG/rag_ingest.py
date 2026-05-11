# rag_ingest.py
import hashlib
from RAG.rag_setup import embedder, collection

def ingest_article(url: str, title: str, content: str, source: str):
    chunks = chunk_text(content, chunk_size=200, overlap=30)
    
    added = 0
    for i, chunk in enumerate(chunks):
        # Bỏ qua chunk quá ngắn
        if len(chunk.split()) < 20:
            continue

        doc_id = hashlib.md5(f"{url}_{i}".encode()).hexdigest()

        # ✅ Fix: kiểm tra đúng cách
        existing = collection.get(ids=[doc_id])
        if existing["ids"]:
            continue

        embedding = embedder.encode(chunk).tolist()
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "url": url,
                "title": title,
                "source": source,
                "chunk_index": i
            }]
        )
        added += 1

    print(f"  → Added {added} chunks từ {source}")

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks