"""
Pipeline kiểm tra thuốc giả
Flow: Ảnh → Siamese + FAISS → Color Validator → OCR Validator → Kết luận
Verdict: CÓ THỂ THẬT / NGHI GIẢ / KHÔNG CÓ TRONG DB
"""
import os
import sys
import warnings
import logging

# Tắt log nhiễu từ PaddleOCR / PaddleX
os.environ.update({
    'GLOG_minloglevel': '2',
    'PADDLE_DISABLE_PROFILER': '1',
    'PADDLEX_LOG_LEVEL': 'CRITICAL',
    'PADDLE_PDLL_PRINT_LOG': '0',
    'FLAGS_enable_pir_api': '0',
    'FLAGS_use_mkldnn': '0',
})
warnings.filterwarnings("ignore")
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddlex').setLevel(logging.ERROR)

import re
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from PIL import Image
from torchvision import transforms, models
from difflib import SequenceMatcher

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ── Cấu hình ──────────────────────────────────────────────────────────────────
EMBED_DIM         = 256
MODEL_PATH        = r"models\siamese_best.pth"
INDEX_PATH        = "faiss.index"
META_PATH         = "faiss_meta.json"

SIAMESE_THRESHOLD = 0.5
COLOR_THRESHOLD   = 0.20
TEXT_THRESHOLD    = 0.70
TOP_K             = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Suppressor stdout/stderr (dùng khi khởi tạo EasyOCR) ─────────────────────
class SuppressOutput:
    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, 'w')

    def __exit__(self, *_):
        sys.stdout.close()
        sys.stdout, sys.stderr = self._stdout, self._stderr


# ── Định nghĩa model ──────────────────────────────────────────────────────────
class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone      = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return F.normalize(self.projector(x), p=2, dim=1)


class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.net = EmbeddingNet(embed_dim)

    def forward(self, x1, x2):
        return self.net(x1), self.net(x2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Khởi tạo model, FAISS index và metadata ───────────────────────────────────
def _load_resources():
    ck    = torch.load(MODEL_PATH, map_location=device)
    model = SiameseNetwork(ck.get('embed_dim', EMBED_DIM)).to(device)
    model.load_state_dict(ck['model_state'])
    model.eval()

    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"✅ Siamese : epoch {ck['epoch']}")
    print(f"✅ FAISS   : {index.ntotal} vectors")
    return model, index, metadata


siamese, faiss_index, meta = _load_resources()

# Preprocessing chuẩn ImageNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Lazy-load EasyOCR ─────────────────────────────────────────────────────────
_ocr_engine = None

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            import easyocr
            with SuppressOutput():
                _ocr_engine = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        except ImportError:
            print("⚠️ EasyOCR chưa cài. Hãy gõ: pip install easyocr")
    return _ocr_engine


# ── Bước 1: Siamese + FAISS ───────────────────────────────────────────────────
def search_medicine(image_path: str) -> dict:
    """Tìm kiếm ảnh thuốc trong DB bằng embedding Siamese + FAISS.
    Trả về found=True nếu khoảng cách đến top-match < threshold
    và query nằm trong cluster của class đó.
    """
    img = Image.open(image_path).convert('RGB')
    x   = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = siamese.get_embedding(x).cpu().numpy().astype(np.float32)

    distances, indices = faiss_index.search(query_emb, TOP_K)
    top_dist = float(distances[0][0])
    top_idx  = int(indices[0][0])
    top_class = meta[top_idx]['class_name']

    # Tính centroid và độ phân tán nội-class để kiểm tra membership
    class_paths = [m['path'] for m in meta if m['class_name'] == top_class]
    class_embs  = []
    for p in class_paths[:10]:
        try:
            img_c = Image.open(p).convert('RGB')
            x_c   = image_transform(img_c).unsqueeze(0).to(device)
            with torch.no_grad():
                class_embs.append(siamese.get_embedding(x_c).cpu().numpy())
        except Exception:
            continue

    if class_embs:
        class_embs    = np.vstack(class_embs).astype(np.float32)
        centroid      = class_embs.mean(axis=0, keepdims=True)
        dist_centroid = float(np.linalg.norm(query_emb - centroid))
        intra_std     = float(np.std([
            np.linalg.norm(class_embs[i] - class_embs[j])
            for i in range(len(class_embs))
            for j in range(i + 1, len(class_embs))
        ])) if len(class_embs) > 1 else 0.1
        intra_std = max(intra_std, 0.05)
    else:
        dist_centroid = top_dist
        intra_std     = 0.05

    is_member = dist_centroid < (intra_std * 5)

    return {
        "found"        : top_dist < SIAMESE_THRESHOLD and is_member,
        "reject_reason": ("Distance > Threshold" if top_dist >= SIAMESE_THRESHOLD
                          else ("Out of Cluster" if not is_member else "")),
        "top_match"    : top_class,
        "top_path"     : meta[top_idx]['path'],
        "distance"     : round(top_dist, 4),
        "dist_centroid": round(dist_centroid, 4),
        "intra_std"    : round(intra_std, 4),
    }


# ── Bước 2A: Color Validator ──────────────────────────────────────────────────
def color_similarity(path1: str, path2: str, bins: int = 64) -> float:
    """So sánh histogram HSV vùng trung tâm của 2 ảnh.
    Dùng Correlation coefficient — càng gần 1 càng giống màu.
    """
    def center_hsv_hist(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            return np.zeros((bins * bins,), dtype=np.float32)
        h, w  = img.shape[:2]
        cx, cy = w // 2, h // 2
        cw, ch = w // 4, h // 4
        crop  = img[cy - ch: cy + ch, cx - cw: cx + cw]
        crop  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist  = cv2.calcHist([crop], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    h1    = center_hsv_hist(path1).astype(np.float32)
    h2    = center_hsv_hist(path2).astype(np.float32)
    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return round(float(score), 4)


# ── Bước 2B: OCR Validator ────────────────────────────────────────────────────
LOT_PATTERNS = [
    r'\bLOT[:\s#]?\s*([A-Z0-9\-]{4,15})\b',
    r'\bBATCH[:\s#]?\s*([A-Z0-9\-]{4,15})\b',
    r'\bL[:\s]?\s*([A-Z0-9\-]{5,12})\b',
    r'\b([A-Z]{1,3}[0-9]{5,8})\b',
]


def ocr_extract(image_path: str) -> dict:
    """Trích xuất text và mã lô (LOT/BATCH) từ ảnh bằng EasyOCR."""
    engine = get_ocr()
    if engine is None:
        return {"text": "", "lot_number": None}

    try:
        abs_path = os.path.abspath(image_path)
        result   = engine.readtext(abs_path)
        if not result:
            return {"text": "", "lot_number": None}

        # Chỉ giữ các token có confidence > 20%
        lines    = [text for _, text, prob in result if prob > 0.2]
        raw_text = " ".join(lines)
        upper    = raw_text.upper()

        print(f"   [OCR] '{raw_text}'")

        lot_number = None
        for pattern in LOT_PATTERNS:
            match = re.search(pattern, upper)
            if match:
                lot_number = match.group(len(match.groups()))
                break

        return {"text": raw_text, "lot_number": lot_number}

    except Exception as e:
        print(f"   [OCR ERROR] {e}")
        return {"text": "", "lot_number": None, "error": str(e)}


def text_similarity(db_name: str, ocr_text: str) -> float:
    """So khớp tên thuốc trong DB với text OCR đọc được từ ảnh.
    Trả về 1.0 ngay nếu từ khóa đầu tiên của tên DB xuất hiện trong OCR text.
    """
    if not ocr_text or not db_name:
        return 0.0

    db_lower  = db_name.lower().strip()
    ocr_lower = ocr_text.lower().strip()
    keyword   = db_lower.split()[0] if db_lower else ""

    if keyword and keyword in ocr_lower:
        return 1.0

    return round(SequenceMatcher(None, db_lower, ocr_lower).ratio(), 4)


# ── Pipeline chính ────────────────────────────────────────────────────────────
def run_pipeline(image_path: str) -> dict:
    print(f"\n{'=' * 58}\nẢnh: {image_path}\n{'=' * 58}")
    result = {"image": image_path}

    # Bước 1: Siamese + FAISS
    print("\nBước 1: Siamese + FAISS...")
    search = search_medicine(image_path)
    result['siamese'] = search
    print(f"   dist_centroid : {search['dist_centroid']}")
    print(f"   intra_std     : {search['intra_std']}")
    print(f"   Top match     : {search['top_match']}")
    print(f"   Distance      : {search['distance']} ({'IN DB' if search['found'] else 'NOT IN DB'})")

    if not search['found']:
        result.update({
            'verdict'    : 'KHÔNG CÓ TRONG DB',
            'explanation': 'Khoảng cách đến tâm class quá xa hoặc vượt ngoài Threshold.',
            'flags'      : [],
        })
        _print_result(result)
        return result

    # Bước 2A: Color Validator
    print("\nBước 2A: Color Validator...")
    color_score = color_similarity(image_path, search['top_path'])
    color_ok    = color_score >= COLOR_THRESHOLD
    result['color_score'] = color_score
    print(f"   Score: {color_score} ({'OK' if color_ok else 'FAIL'} | threshold={COLOR_THRESHOLD})")

    # Bước 2B: OCR Validator
    print("\nBước 2B: OCR Validator...")
    ocr_result  = ocr_extract(image_path)
    text_score  = text_similarity(search['top_match'], ocr_result['text'])
    text_ok     = text_score >= TEXT_THRESHOLD
    result['text_score'] = text_score
    result['lot_number'] = ocr_result.get('lot_number')
    print(f"   Score: {text_score} ({'OK' if text_ok else 'FAIL'} | threshold={TEXT_THRESHOLD})")
    if ocr_result['lot_number']:
        print(f"   Mã lô: {ocr_result['lot_number']} (tham khảo)")

    # Smart override: bỏ qua lỗi màu nếu OCR và Siamese đều rất khớp
    if text_ok and text_score == 1.0 and search['distance'] < 0.3 and not color_ok:
        print("   [OVERRIDE] Bỏ qua lỗi màu sắc do OCR và Siamese quá khớp.")
        color_ok = True

    # Bước 3: Kết luận
    flags = []
    if not color_ok:
        flags.append(f"Màu sắc bất thường (score={color_score} < {COLOR_THRESHOLD})")
    if not text_ok:
        flags.append(f"Text không khớp (score={text_score} < {TEXT_THRESHOLD})")

    result['flags'] = flags
    if not flags:
        result.update({
            'verdict'    : 'CÓ THỂ THẬT',
            'explanation': f"Khớp với {search['top_match']} — màu sắc và text đều hợp lệ.",
        })
    else:
        result.update({
            'verdict'    : 'NGHI GIẢ',
            'explanation': " | ".join(flags),
        })

    _print_result(result)
    return result


def _print_result(result: dict):
    print(f"\n{'─' * 58}")
    print(f"VERDICT    : {result['verdict']}")
    print(f"GIẢI THÍCH : {result['explanation']}")
    for flag in result.get('flags', []):
        print(f"  ⚠ {flag}")
    print(f"{'─' * 58}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "Acretin-0-05_-Cream-30G-Jamjoom.jpg"
    run_pipeline(path)