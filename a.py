import cv2
import numpy as np
import os
import torch
import re
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ---------------- CONFIG ----------------
IMAGE_PATH = "test.jpg"
OUTPUT_DIR = "outputs"
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

TROCR_MODEL = "microsoft/trocr-large-handwritten"

# ---------------- HELPERS ----------------
def is_garbage(text):
    t = text.strip()
    if len(t) < 3:
        return True
    if re.fullmatch(r"[#\.\-_,:;]+", t):
        return True
    if sum(c.isalnum() for c in t) / max(len(t), 1) < 0.3:
        return True
    return False

def auto_rotate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return img

    angles = [(l[0][1] - np.pi / 2) * 180 / np.pi for l in lines[:30]]
    if not angles:
        return img

    angle = np.median(angles)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def enhance_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    return cv2.cvtColor(enh, cv2.COLOR_GRAY2RGB)

def crop_polygon(image, poly):
    pts = np.array(poly, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype="float32")

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW < 10 or maxH < 10:
        return None

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def box_center_y(box):
    return np.mean([p[1] for p in box])

# ---------------- LOAD MODELS ----------------
print("🔄 Loading PaddleOCR...")
ocr = PaddleOCR(lang="en", use_textline_orientation=True)

print("🔄 Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# ---------------- LOAD IMAGE ----------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"❌ Image not found: {IMAGE_PATH}")

img = auto_rotate(img)
debug_img = img.copy()
cv2.imwrite(os.path.join(OUTPUT_DIR, "rotated.jpg"), img)

# ---------------- DETECT BOXES ----------------
print("🔍 Detecting text boxes...")
results = ocr.predict(img)

boxes = []
for page in results:
    for poly in page["dt_polys"]:
        boxes.append(poly)

if not boxes:
    raise RuntimeError("❌ No text detected")

boxes = sorted(boxes, key=box_center_y)

# ---------------- DRAW DEBUG BOXES ----------------
for poly in boxes:
    pts = np.array(poly, np.int32)
    cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_boxes.jpg"), debug_img)

# ---------------- GROUP INTO PARAGRAPHS ----------------
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
seen_lines = set()
crop_id = 0

for pid, para in enumerate(paragraphs):
    para_text = []

    for poly in para:
        crop = crop_polygon(img, poly)
        if crop is None:
            continue

        crop = enhance_crop(crop)

        crop_path = os.path.join(CROP_DIR, f"crop_{crop_id}.jpg")
        cv2.imwrite(crop_path, crop)
        crop_id += 1

        pil_img = Image.fromarray(crop).convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=128)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        norm = re.sub(r"\s+", " ", text.lower())

        if norm in seen_lines:
            continue

        if not is_garbage(text):
            para_text.append(text)
            seen_lines.add(norm)

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

print("\n✅ Saved OCR text to:", out_path)
print("🟩 Saved cropped words to:", CROP_DIR)
print("🖼 Debug image with boxes: outputs/debug_boxes.jpg")