"""
Pipeline phân loại tin y tế giả/thật
Flow: Text → PhoBERT (lọc y tế) → TF-IDF + SVM (Thật/Giả) → RAG + LLM → Kết luận
"""
import os
import time
import json

import torch
import joblib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq
import groq
from dotenv import load_dotenv

from RAG.rag_retrieve import retrieve_evidence

# ── Cấu hình ──────────────────────────────────────────────────────────────────
PHOBERT_DIR   = r"models\PhoBERT_Model"
SVM_PATH      = r"models\svm_fake_model.pkl"
TFIDF_PATH    = r"models\tfidf.pkl"
GROQ_MODEL    = "llama-3.3-70b-versatile"
RAG_TOP_K     = 6
MAX_API_RETRY = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load models ───────────────────────────────────────────────────────────────
print("Đang tải mô hình PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_DIR, use_fast=False)
phobert   = AutoModelForSequenceClassification.from_pretrained(PHOBERT_DIR).to(device)
phobert.eval()

print("Đang tải mô hình SVM & TF-IDF...")
svm_model  = joblib.load(SVM_PATH)
vectorizer = joblib.load(TFIDF_PATH)

load_dotenv()
groq_client = Groq(api_key=os.getenv("groq_api_key"))


# ── LLM + RAG ─────────────────────────────────────────────────────────────────
def generate_verdict(text: str, prob_fake: float, valid_evidence: list[dict]) -> dict:
    """Gửi bài viết + evidence hợp lệ lên Groq/LLaMA để tổng hợp kết luận cuối."""
    
    # KỊCH BẢN 1: Bị chặn ngay từ vòng RAG (không có chunk nào qua được ngưỡng)
    if not valid_evidence:
        return {
            "verdict" : "CHƯA RÕ",
            "explanation" : "Không tìm thấy bằng chứng y khoa nào đủ độ tin cậy trong cơ sở dữ liệu để xác minh thông tin này.",
            "cited_sources": [],
        }

    evidence_text = "\n\n".join(f"[{e['source']}] {e['content']}" for e in valid_evidence)

    if prob_fake > 0.85:
        instruction = "Hệ thống AI đánh giá cao khả năng đây là TIN GIẢ. Hãy đối chiếu với nguồn tài liệu bên dưới để chỉ ra điểm sai lệch."
    elif prob_fake < 0.15:
        instruction = "Hệ thống AI đánh giá đây là TIN THẬT. Hãy kiểm chứng độ chính xác dựa vào nguồn tài liệu cung cấp."
    else:
        instruction = "Hệ thống AI chưa chắc chắn. Hãy phân tích và đưa ra kết luận dựa trên nguồn tài liệu."

    # KỊCH BẢN 2: Lệnh rào ép LLM không được tự "ảo giác" (Hallucination)
    prompt = f"""{instruction}

BÀI VIẾT CẦN KIỂM CHỨNG: {text}

TÀI LIỆU RAG CUNG CẤP:
{evidence_text}

LUẬT BẮT BUỘC (QUAN TRỌNG):
1. CHỈ sử dụng thông tin từ "TÀI LIỆU RAG CUNG CẤP" để đưa ra kết luận.
2. Nếu tài liệu cung cấp KHÔNG chứa thông tin trực tiếp để xác nhận hoặc bác bỏ bài viết (ví dụ: tài liệu nói về bệnh khác, mẹo khác), bạn BẮT BUỘC phải kết luận là "CHƯA RÕ".
3. Tuyệt đối KHÔNG tự suy diễn bằng kiến thức nền của bạn.

Bạn bắt buộc phải trả về định dạng JSON hợp lệ với cấu trúc sau:
{{
  "verdict": "THẬT" hoặc "GIẢ" hoặc "CHƯA RÕ",
  "explanation": "Giải thích 2-3 câu bằng tiếng Việt",
  "cited_sources": ["url1", "url2"]
}}"""

    for attempt in range(MAX_API_RETRY):
        try:
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=GROQ_MODEL,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except groq.RateLimitError:
            wait_time = (attempt + 1) * 10
            print(f"\n[!] Groq Rate Limit. Đợi {wait_time}s trước khi thử lại...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"\n[!] Lỗi API: {e}")
            break

    return {
        "verdict" : "LỖI HỆ THỐNG",
        "explanation" : "Không thể tạo kết luận do lỗi kết nối với Groq API.",
        "cited_sources": [],
    }

# ── Dự đoán chính ─────────────────────────────────────────────────────────────
def predict_news(text: str) -> dict:
    # Bước 1: PhoBERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        is_health = phobert(**inputs).logits.argmax(dim=1).item()

    if is_health == 0:
        return {"label": "KHÔNG PHẢI TIN Y TẾ", "prob_real": None, "prob_fake": None, "rag": None}

    # Bước 2: SVM
    tfidf_vec  = vectorizer.transform([text])
    probs      = svm_model.predict_proba(tfidf_vec)[0]
    prob_real, prob_fake = probs[0], probs[1]
    label      = "TIN Y TẾ GIẢ" if prob_fake > prob_real else "TIN Y TẾ THẬT"

    # Bước 3: RAG tìm evidence có kèm Similarity Score
    # Lưu ý: Cần đảm bảo hàm retrieve_evidence trả về dict có chứa key 'score' (ví dụ: e['score'])
    evidence = retrieve_evidence(text, top_k=RAG_TOP_K)
    
    # BÀN TAY SẮT: Chặn đứng các chunk rác có độ tương đồng thấp (< 0.5)
    THRESHOLD = 0.50 
    valid_evidence = [e for e in evidence if e.get('score', 1.0) >= THRESHOLD]

    rag_result = generate_verdict(text, prob_fake, valid_evidence)

    return {
        "label"    : label,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "rag"      : {
            "verdict"       : rag_result.get("verdict"),
            "explanation"   : rag_result.get("explanation"),
            "cited_sources" : rag_result.get("cited_sources", []),
            "evidence_count": len(valid_evidence), # Chỉ đếm các chunk hợp lệ
        },
    }


# ── Visualize kết quả ─────────────────────────────────────────────────────────
def plot_result(text: str, result: dict):
    """Vẽ biểu đồ tròn xác suất Thật/Giả. Hiển thị 3s rồi tự đóng."""
    if result["prob_real"] is None:
        return

    plt.figure(figsize=(6, 6))
    plt.pie(
        [result["prob_real"] * 100, result["prob_fake"] * 100],
        labels=["Tin Thật", "Tin Giả"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4CAF50", "#F44336"],
    )
    plt.axis("equal")
    plt.title(f"Kết quả phân loại\n\"{text[:60]}...\"")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("HỆ THỐNG PHÂN LOẠI TIN Y TẾ (Powered by Groq & LLaMA 3)")
    print("=" * 50)

    while True:
        text = input("\nNhập tin (gõ 'exit' để thoát): ").strip()
        if text.lower() == "exit":
            break

        result = predict_news(text)
        print(f"\n{'=' * 50}")
        print(f"  {result['label']}")

        if result["prob_real"] is not None:
            print(f"  → Tin Thật : {result['prob_real'] * 100:.2f}%")
            print(f"  → Tin Giả  : {result['prob_fake'] * 100:.2f}%")

        if result["rag"]:
            rag = result["rag"]
            print(f"\n--- NHẬN ĐỊNH TỪ HỆ THỐNG RAG (LLAMA 3) ---")
            print(f"VERDICT    : {rag['verdict']}")
            print(f"GIẢI THÍCH : {rag['explanation']}")
            if rag["cited_sources"]:
                print("NGUỒN DẪN :")
                for url in rag["cited_sources"]:
                    print(f"  - {url}")
            print(f"(Dựa trên {rag['evidence_count']} đoạn tài liệu liên quan)")

        plot_result(text, result)