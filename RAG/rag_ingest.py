# RAG/rag_ingest.py
"""
Ingest bài báo vào ChromaDB.
- chunk_text()    : chia nội dung thành chunks có overlap
- ingest_article(): ingest 1 bài, dedup theo URL (kiểm tra chunk_0)
- ingest_from_file(): ingest toàn bộ từ articles.json (CLI helper)
"""

import hashlib
import json
from pathlib import Path

from RAG.rag_setup import embedder, collection


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """
    Chia text thành các chunk theo số từ với sliding window overlap.
    Bỏ chunk < 20 từ (thường là fragment cuối bài ngắn).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Core ingest ───────────────────────────────────────────────────────────────
def ingest_article(
    url: str,
    title: str,
    content: str,
    source: str,
    author: str = "Không rõ",
    published_at: str = "",
    crawled_at: str = "",
) -> int:
    """
    Chunk và embed 1 bài báo, lưu vào ChromaDB.

    Dedup theo URL: kiểm tra chunk_0 của bài. Nếu đã có → skip toàn bài.
    Title được prepend vào content trước khi chunk để chunk đầu tiên
    luôn chứa keyword quan trọng nhất.

    Returns: số chunks đã add (0 nếu đã tồn tại hoặc lỗi)
    """
    if not url or not content:
        return 0

    url_hash = hashlib.md5(url.encode()).hexdigest()

    # ── Dedup: check chunk_0 thay vì từng chunk riêng lẻ ─────────────────────
    # Tránh trường hợp ingest dở → lần sau skip sai
    chunk_0_id = f"{url_hash}_chunk0"
    try:
        existing = collection.get(ids=[chunk_0_id])
        if existing["ids"]:
            return 0  # Bài đã có đầy đủ → skip
    except Exception:
        pass  # Collection mới chưa có gì → tiếp tục

    # ── Chuẩn bị text: title + content ────────────────────────────────────────
    full_text = f"{title.strip()}\n\n{content.strip()}"
    chunks = chunk_text(full_text)

    if not chunks:
        return 0

    # ── Embed và add ──────────────────────────────────────────────────────────
    ids        = [f"{url_hash}_chunk{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    metadatas  = [
        {
            "url":          url,
            "title":        title,
            "source":       source,
            "author":       author,
            "published_at": published_at,
            "crawled_at":   crawled_at,
            "chunk_index":  i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


# ── CLI: ingest từ articles.json ──────────────────────────────────────────────
def ingest_from_file(articles_path: str = "articles.json") -> None:
    """
    Đọc articles.json và ingest tất cả bài chưa có trong DB.
    Dùng khi muốn seed DB thủ công lần đầu.

    Usage:
        python -m RAG.rag_ingest                     # đọc ./articles.json
        python -m RAG.rag_ingest path/to/file.json   # chỉ định file
    """
    path = Path(articles_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {articles_path}")

    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"📂 Đọc {len(articles)} bài từ {articles_path}")
    ok = skip = err = 0

    for art in articles:
        try:
            added = ingest_article(
                url=art.get("url", ""),
                title=art.get("title", ""),
                content=art.get("content", ""),
                source=art.get("source", "Unknown"),
                author=art.get("author", "Không rõ"),
                published_at=art.get("published_at", ""),
                crawled_at=art.get("crawled_at", ""),
            )
            if added > 0:
                ok += 1
                print(f"  [OK]   +{added} chunks | {art.get('title', '')[:60]}")
            else:
                skip += 1
        except Exception as e:
            err += 1
            print(f"  [ERR]  {art.get('url', '')} — {e}")

    print(f"\n✅ Xong: {ok} ingested | {skip} skipped (trùng) | {err} lỗi")
    print(f"📦 Tổng DB: {collection.count()} chunks")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "articles.json"
    ingest_from_file(path)