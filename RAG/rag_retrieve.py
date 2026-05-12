# RAG/rag_retrieve.py
"""
Retrieve evidence từ ChromaDB cho RAG pipeline.

BUG ĐÃ FIX:
  Trước: query n_results=top_k → filter → còn lại ít hơn top_k
  Sau:   query n_results=top_k*2 → filter → slice[:top_k]
  → LLM luôn nhận đủ số evidence như pipeline yêu cầu.
"""

from RAG.rag_setup import embedder, collection


def retrieve_evidence(query: str, top_k: int = 5) -> list[dict]:
    """
    Tìm kiếm evidence liên quan đến query trong ChromaDB.

    Args:
        query:  Câu hỏi / bài viết cần kiểm tra
        top_k:  Số evidence muốn nhận về (pipeline điều chỉnh theo confidence)

    Returns:
        List[dict] mỗi phần tử gồm:
            content  : đoạn text gốc
            source   : tên báo nguồn
            url      : link bài gốc
            title    : tiêu đề bài
            score    : cosine similarity (0-1, càng cao càng liên quan)
    """
    # ── Guard: DB còn trống ───────────────────────────────────────────────────
    total_docs = collection.count()
    if total_docs == 0:
        print("  [RAG] ⚠️  ChromaDB chưa có dữ liệu. Chạy rag_ingest.py trước.")
        return []

    # Fetch nhiều hơn để sau filter vẫn đủ top_k
    n_fetch = min(top_k * 2, total_docs)

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_fetch,
    )

    # ── Lọc theo threshold và format ─────────────────────────────────────────
    evidence = []
    for i in range(len(results["ids"][0])):
        # ChromaDB cosine space trả distance → similarity = 1 - distance
        similarity = 1 - results["distances"][0][i]

        print(
            f"  [RAG DEBUG] chunk {i}: sim={similarity:.3f} | "
            f"{results['documents'][0][i][:60]}..."
        )

        if similarity < 0.30:  # threshold: lọc chunk không liên quan
            continue

        meta = results["metadatas"][0][i]
        evidence.append(
            {
                "content": results["documents"][0][i],
                "source":  meta.get("source", "Unknown"),
                "url":     meta.get("url", ""),
                "title":   meta.get("title", ""),
                "score":   round(similarity, 3),
            }
        )

    # Trả đúng top_k sau khi filter
    return evidence[:top_k]