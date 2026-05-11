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
RAG_TOP_K     = 3
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
def generate_verdict(text: str, prob_fake: float, evidence: list[dict]) -> dict:
    """Gửi bài viết + evidence lên Groq/LLaMA để tổng hợp kết luận cuối.
    Prompt thay đổi theo confidence của SVM để định hướng LLM.
    """
    if not evidence:
        return {
            "rag_verdict" : None,
            "explanation" : "Không tìm thấy bằng chứng liên quan trong cơ sở dữ liệu.",
            "cited_sources": [],
        }

    evidence_text = "\n\n".join(f"[{e['source']}] {e['content']}" for e in evidence)

    # Định hướng LLM dựa trên confidence SVM
    if prob_fake > 0.85:
        instruction = "Hệ thống AI đánh giá cao khả năng đây là TIN GIẢ. Dựa vào nguồn uy tín, hãy giải thích ngắn gọn điểm sai lệch."
    elif prob_fake < 0.15:
        instruction = "Hệ thống AI đánh giá đây là TIN THẬT. Dựa vào nguồn uy tín, hãy xác nhận thông tin."
    else:
        instruction = "Hệ thống AI chưa chắc chắn. Hãy phân tích và đưa ra kết luận dựa trên nguồn uy tín."

    prompt = f"""{instruction}

BÀI VIẾT: {text}

NGUỒN UY TÍN LIÊN QUAN:
{evidence_text}

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
        "rag_verdict" : "LỖI HỆ THỐNG",
        "explanation" : "Không thể tạo kết luận do lỗi kết nối với Groq API.",
        "cited_sources": [],
    }


# ── Dự đoán chính ─────────────────────────────────────────────────────────────
def predict_news(text: str) -> dict:
    """Pipeline 3 bước:
    1. PhoBERT: lọc bài không phải tin y tế.
    2. SVM + TF-IDF: phân loại Thật / Giả.
    3. RAG + LLM: tổng hợp bằng chứng và đưa ra kết luận cuối.
    """
    # Bước 1: Kiểm tra có phải tin y tế không
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        is_health = phobert(**inputs).logits.argmax(dim=1).item()

    if is_health == 0:
        return {"label": "KHÔNG PHẢI TIN Y TẾ", "prob_real": None, "prob_fake": None, "rag": None}

    # Bước 2: Phân loại Thật/Giả bằng SVM
    tfidf_vec  = vectorizer.transform([text])
    probs      = svm_model.predict_proba(tfidf_vec)[0]
    prob_real, prob_fake = probs[0], probs[1]
    label      = "TIN Y TẾ GIẢ" if prob_fake > prob_real else "TIN Y TẾ THẬT"

    # Bước 3: RAG tìm evidence → LLM tổng hợp kết luận
    evidence   = retrieve_evidence(text, top_k=RAG_TOP_K)
    rag_result = generate_verdict(text, prob_fake, evidence)

    return {
        "label"    : label,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "rag"      : {
            "verdict"       : rag_result.get("verdict"),
            "explanation"   : rag_result.get("explanation"),
            "cited_sources" : rag_result.get("cited_sources", []),
            "evidence_count": len(evidence),
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