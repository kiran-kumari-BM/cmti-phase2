import cv2
import numpy as np
import os
import torch
import re
import easyocr
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ---------------- CONFIG ----------------
IMAGE_PATH = "test2.jpeg"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TROCR_MODEL = "microsoft/trocr-large-handwritten"

# ---------------- HELPERS ----------------
def is_garbage(text):
    t = text.strip()
    if len(t) < 3:
        return True
    if re.fullmatch(r"[#\.\-_,:;]+", t):
        return True
    if sum(c.isalnum() for c in t) / max(len(t), 1) < 0.25:
        return True
    return False

# ---------------- AUTO ROTATE ----------------
def auto_rotate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        return img
    angles = [(l[0][1] - np.pi/2) * 180 / np.pi for l in lines[:30]]
    if not angles:
        return img
    angle = np.median(angles)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

# ---------------- LOAD MODELS ----------------
print("🔄 Loading OCR engines...")
paddle_ocr = PaddleOCR(lang="en", use_textline_orientation=True)
easy_ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# ---------------- LOAD IMAGE ----------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"❌ Image not found: {IMAGE_PATH}")

img = auto_rotate(img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "rotated.jpg"), img)

# ---------------- DETECT BOXES (PaddleOCR → EasyOCR fallback) ----------------
def detect_boxes(img):
    boxes = []
    try:
        raw = paddle_ocr.predict(img)
        for page in raw:
            for poly in page["dt_polys"]:
                boxes.append(poly)
    except:
        pass
    return boxes

print("🔍 Detecting text boxes...")
boxes = detect_boxes(img)

if not boxes:
    print("⚠ PaddleOCR failed → switching to EasyOCR")
    results = easy_ocr.readtext(img)
    for (bbox, text, conf) in results:
        boxes.append(bbox)

if not boxes:
    raise RuntimeError("❌ No text detected by any OCR engine.")

# ---------------- SORT + GROUP ----------------
def box_center_y(box):
    return np.mean([p[1] for p in box])

boxes = sorted(boxes, key=box_center_y)

paragraphs, current, prev_y = [], [], None
for b in boxes:
    y = box_center_y(b)
    if prev_y is None or abs(y - prev_y) < 45:
        current.append(b)
    else:
        paragraphs.append(current)
        current = [b]
    prev_y = y
if current:
    paragraphs.append(current)

# ---------------- OCR EACH BOX ----------------
final_output = []

for para in paragraphs:
    para_text = []
    for box in para:
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
            para_text.append(text)

    if para_text:
        final_output.append(para_text)

# ---------------- SAVE OUTPUT ----------------
out_path = os.path.join(OUTPUT_DIR, "result.txt")
with open(out_path, "w") as f:
    for i, para in enumerate(final_output):
        f.write(f"\n[Paragraph {i+1}]\n")
        for line in para:
            f.write(line + "\n")

print("\n=========== FINAL OCR OUTPUT ===========")
for i, para in enumerate(final_output):
    print(f"\n[Paragraph {i+1}]")
    for line in para:
        print(line)

print("\n✅ Saved to:", out_path)