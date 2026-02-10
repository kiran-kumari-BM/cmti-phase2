import cv2
import numpy as np
import os
import math
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from spellchecker import SpellChecker

# ---------------- CONFIG ----------------
IMAGE_PATH = "test9.jpeg"
TROCR_MODEL = "microsoft/trocr-base-handwritten"   # faster + stable
USE_MPS = True

# ---------------- LOAD MODELS ----------------
print("🔄 Loading PaddleOCR...")
ocr = PaddleOCR(lang="en", use_textline_orientation=True)

print("🔄 Loading TrOCR...")
device = "mps" if USE_MPS and torch.backends.mps.is_available() else "cpu"
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL).to(device)
model.eval()

spell = SpellChecker()

# ---------------- HELPERS ----------------
def auto_rotate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        return img

    angles = []
    for rho, theta in lines[:20]:
        angle = (theta - np.pi/2) * 180/np.pi
        angles.append(angle)

    median_angle = np.median(angles)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def remove_ruled_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,15,3)

    horizontal = bw.copy()
    vertical = bw.copy()

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))

    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, v_kernel)

    mask = cv2.bitwise_or(horizontal, vertical)
    cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def recognize_crop(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=rgb, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def smart_spell_fix(text):
    words = text.split()
    fixed = []
    for w in words:
        if w.lower() in spell:
            fixed.append(w)
        else:
            fixed.append(spell.correction(w) or w)
    return " ".join(fixed)


def group_paragraphs(lines, gap=40):
    paragraphs = []
    current = []
    last_y = None

    for y, text, conf in lines:
        if last_y is None or abs(y - last_y) < gap:
            current.append((text, conf))
        else:
            paragraphs.append(current)
            current = [(text, conf)]
        last_y = y

    if current:
        paragraphs.append(current)
    return paragraphs


# ---------------- MAIN PIPELINE ----------------
if not os.path.exists(IMAGE_PATH):
    raise ValueError(f"❌ Image not found: {IMAGE_PATH}")

img = cv2.imread(IMAGE_PATH)
img = auto_rotate(img)
img = remove_ruled_lines(img)

print("🔍 Detecting text boxes...")
raw = ocr.ocr(img)[0]

lines = []

for r in raw:
    try:
        box = None
        conf = 1.0

        # PaddleOCR v3 dict format
        if isinstance(r, dict) and "points" in r:
            box = np.array(r["points"], dtype=np.float32).astype(int)
            conf = float(r.get("score", 1.0))

        # PaddleOCR classic format
        elif isinstance(r, (list, tuple)) and len(r) >= 2 and isinstance(r[0], (list, tuple)):
            box = np.array(r[0], dtype=np.float32).astype(int)
            if isinstance(r[1], (list, tuple)) and len(r[1]) == 2:
                conf = float(r[1][1])

        else:
            continue

        x, y, w, h = cv2.boundingRect(box)
        if w < 40 or h < 20:
            continue

        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        text = recognize_crop(crop)
        text = smart_spell_fix(text)

        if len(text.strip()) > 1:
            lines.append((y, text.strip(), conf))

    except Exception as e:
        print("⚠️ Skipped one bad box:", e)
        continue

lines.sort(key=lambda x: x[0])
paragraphs = group_paragraphs(lines)

# ---------------- OUTPUT ----------------
print("\n\n📄 OCR OUTPUT\n")

for i, para in enumerate(paragraphs, 1):
    print(f"[Paragraph {i}]")
    for text, conf in para:
        print(f"{text}   (conf: {conf:.2f})")
    print()