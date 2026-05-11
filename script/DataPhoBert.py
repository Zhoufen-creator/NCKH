import pandas as pd
import re
import os

# CẤU HÌNH
MEDICAL_FILE = "Dataset_VIE.csv"         # file tin y tế
NON_MEDICAL_FILE = "Dataset_articles.csv" # file không phải y tế
OUTPUT_FILE = "health_classification_dataset.csv"

# HÀM HỖ TRỢ
def clean_text(text):
    """Làm sạch text cơ bản"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_text_column(df):
    """
    Tự động tìm cột chứa nội dung văn bản
    """
    candidates = ["text", "content", "contents", "body", "article", "news"]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    raise ValueError(f"Không tìm thấy cột nội dung trong file. Các cột hiện có: {list(df.columns)}")


# LOAD DATA
print("Đọc dữ liệu...")

df_med = pd.read_csv(MEDICAL_FILE, encoding="utf-8")
df_non = pd.read_csv(NON_MEDICAL_FILE, encoding="utf-8")
# tìm cột text
med_text_col = find_text_column(df_med)
non_text_col = find_text_column(df_non)
print(f"File y tế dùng cột: {med_text_col}")
print(f"File không y tế dùng cột: {non_text_col}")
# chuẩn hóa cột text
df_med = df_med.rename(columns={med_text_col: "text"})
df_non = df_non.rename(columns={non_text_col: "text"})

# GÁN NHÃN
df_med["is_health"] = 1
df_non["is_health"] = 0

# GIỮ CỘT CẦN THIẾT
df_med = df_med[["text", "is_health"]]
df_non = df_non[["text", "is_health"]]

# GỘP DATASET
df_all = pd.concat([df_med, df_non], ignore_index=True)

# LÀM SẠCH
df_all["text"] = df_all["text"].astype(str).apply(clean_text)
# bỏ text quá ngắn
before = len(df_all)
df_all = df_all[df_all["text"].str.len() > 20]
after = len(df_all)
print(f"Bỏ {before - after} dòng vì text quá ngắn")
# shuffle dữ liệu
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# LƯU FILE
df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("HOÀN THÀNH")
print("File đã tạo:", OUTPUT_FILE)
print("Thống kê nhãn:")
print(df_all["is_health"].value_counts())
