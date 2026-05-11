import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "phobert_health_model"
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model đã train
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=False
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)
model.to(device)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return pred

while True:
    text = input("\nNhập văn bản (gõ exit để thoát): ")
    if text.lower() == "exit":
        break

    result = predict(text)
    print("Kết quả:", "Y TẾ" if result == 1 else "KHÔNG Y TẾ")
