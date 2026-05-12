# RAG/rag_server.py
"""
FastAPI server nhận bài báo từ n8n và ingest vào ChromaDB.

Khởi động:
    uvicorn RAG.rag_server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /ingest      — nhận batch bài từ n8n  (main endpoint)
    GET  /status      — số chunk trong DB
    GET  /health      — kiểm tra server sống
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from RAG.rag_ingest import ingest_article
from RAG.rag_setup import collection

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Ingest Server",
    description="Nhận bài báo y tế từ n8n → ChromaDB",
    version="2.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class Article(BaseModel):
    source:       str
    title:        str
    content:      str
    author:       Optional[str] = "Không rõ"
    published_at: Optional[str] = ""
    url:          str
    crawled_at:   Optional[str] = ""

    @field_validator("content")
    @classmethod
    def content_must_have_text(cls, v: str) -> str:
        if not v or len(v.strip()) < 50:
            raise ValueError("content quá ngắn (< 50 ký tự) — bỏ qua bài này")
        return v.strip()

    @field_validator("url")
    @classmethod
    def url_must_be_valid(cls, v: str) -> str:
        if not v or not v.startswith("http"):
            raise ValueError(f"url không hợp lệ: {v!r}")
        return v


class IngestRequest(BaseModel):
    articles: list[Article]


class IngestResponse(BaseModel):
    total_received: int
    total_ingested: int
    total_skipped:  int
    skipped_urls:   list[str]
    errors:         list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/status")
def status():
    """Trả về số chunks hiện có trong ChromaDB."""
    try:
        count = collection.count()
        return {
            "status":           "ok",
            "chunk_count":      count,
            "collection_name":  collection.name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Nhận list bài báo từ n8n và ingest vào ChromaDB.

    n8n gửi body: { "articles": [ {...}, {...} ] }

    Response được Log Result node trong n8n đọc:
        total_ingested, total_skipped, errors
    """
    if not request.articles:
        raise HTTPException(status_code=400, detail="Danh sách articles rỗng")

    ingested     = 0
    skipped_urls = []
    errors       = []

    for art in request.articles:
        try:
            added = ingest_article(
                url=art.url,
                title=art.title,
                content=art.content,
                source=art.source,
                author=art.author or "Không rõ",
                published_at=art.published_at or "",
                crawled_at=art.crawled_at or datetime.now().isoformat(),
            )

            if added > 0:
                ingested += 1
                print(f"  [INGEST] +{added} chunks | {art.title[:60]}")
            else:
                skipped_urls.append(art.url)
                print(f"  [SKIP]   đã có: {art.url}")

        except Exception as e:
            errors.append(f"{art.url}: {e}")
            print(f"  [ERR]    {art.url} — {e}")

    print(
        f"\n[/ingest] nhận={len(request.articles)} | "
        f"ingested={ingested} | skipped={len(skipped_urls)} | errors={len(errors)}"
    )

    return IngestResponse(
        total_received=len(request.articles),
        total_ingested=ingested,
        total_skipped=len(skipped_urls),
        skipped_urls=skipped_urls,
        errors=errors,
    )