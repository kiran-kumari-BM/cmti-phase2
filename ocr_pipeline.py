import cv2
import numpy as np
import torch
import re
import easyocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ---------------- LOAD MODELS (Load Once Only) ----------------
print("🔄 Loading OCR models... Please wait...")

paddle_ocr = PaddleOCR(lang="en", use_textline_orientation=True)
easy_ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

TROCR_MODEL = "microsoft/trocr-large-handwritten"
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("✅ Models Loaded Successfully!")


# ---------------- HELPER FUNCTION ----------------
def is_garbage(text):
    t = text.strip()
    if len(t) < 3:
        return True
    if re.fullmatch(r"[#\.\-_,:;]+", t):
        return True
    if sum(c.isalnum() for c in t) / max(len(t), 1) < 0.25:
        return True
    return False


# ---------------- MAIN OCR FUNCTION ----------------
def run_ocr(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return ["❌ Image not found"]

    boxes = []

    # -------- PaddleOCR Detection --------
    try:
        raw = paddle_ocr.predict(img)
        for page in raw:
            for poly in page["dt_polys"]:
                boxes.append(poly)
    except:
        pass

    # -------- EasyOCR Fallback --------
    if not boxes:
        results = easy_ocr.readtext(img)
        for (bbox, text, conf) in results:
            boxes.append(bbox)

    if not boxes:
        return ["❌ No text detected"]

    final_output = []

    # -------- Recognize Each Box using TrOCR --------
    for box in boxes:
        pts = np.array(box).astype(int)
        x, y, w, h = cv2.boundingRect(pts)
        crop = img[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        pil_img = Image.fromarray(crop).convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=128)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        if not is_garbage(text):
            final_output.append(text)

    if not final_output:
        return ["⚠ Text detected but could not extract properly"]

    return final_output