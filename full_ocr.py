import cv2
import os
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

IMAGE_PATH = "test6.jpeg"
OUT_DIR = "outputs"
CROP_DIR = os.path.join(OUT_DIR, "crops")
os.makedirs(CROP_DIR, exist_ok=True)

# ------------------ DBNet Detector ------------------
print("🚀 Loading DBNet (PaddleOCR)...")
ocr_det = PaddleOCR(
    lang="en",
    use_textline_orientation=False,
    text_det_box_thresh=0.2,
    text_det_unclip_ratio=1.2
)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("❌ Image not found")

H, W = img.shape[:2]

result = ocr_det.ocr(IMAGE_PATH)
polys = result[0]["dt_polys"]
print(f"✅ Detected {len(polys)} text boxes")

def poly_to_rect(poly):
    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    return [x, y, x + w, y + h, (y + y + h) // 2, h]

rects = [poly_to_rect(p) for p in polys]

# ------------------ CLUSTER BY ROW ------------------
rects.sort(key=lambda r: r[4])  # sort by y-center

rows = []
ROW_GAP = 25  # <-- tune this if lines are closer/farther in your notebook

for r in rects:
    x1, y1, x2, y2, cy, h = r
    placed = False

    for row in rows:
        _, ry1, _, ry2, rcy, rh = row["bbox"]
        if abs(cy - rcy) < max(ROW_GAP, rh * 0.7):
            row["items"].append(r)
            placed = True
            break

    if not placed:
        rows.append({"bbox": r, "items": [r]})

# ------------------ MERGE EACH ROW INTO CLEAN LINE BOX ------------------
line_boxes = []

for row in rows:
    xs1 = [r[0] for r in row["items"]]
    ys1 = [r[1] for r in row["items"]]
    xs2 = [r[2] for r in row["items"]]
    ys2 = [r[3] for r in row["items"]]

    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)

    # padding but keep inside image
    pad = 5
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad)
    y2 = min(H, y2 + pad)

    line_boxes.append([x1, y1, x2, y2])

line_boxes.sort(key=lambda b: b[1])

# ------------------ DRAW DEBUG ------------------
dbg = img.copy()
for (x1, y1, x2, y2) in line_boxes:
    cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imwrite(os.path.join(OUT_DIR, "lines_ruled_pages.jpg"), dbg)
print("🖼 Saved debug image: outputs/lines_ruled_pages.jpg")

# ------------------ CROP LINES ------------------
line_images = []
for i, (x1, y1, x2, y2) in enumerate(line_boxes):
    crop = img[y1:y2, x1:x2]
    path = os.path.join(CROP_DIR, f"line_{i}.png")
    cv2.imwrite(path, crop)
    line_images.append(path)

print(f"✂️ Cropped {len(line_images)} clean lines")

# ------------------ TrOCR Recognizer ------------------
print("🧠 Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.eval()

final_text = []

for path in line_images:
    image = Image.open(path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    final_text.append(text)

# ------------------ OUTPUT ------------------
print("\n📜 FINAL OCR OUTPUT:\n")
for i, t in enumerate(final_text):
    print(f"{i+1}. {t}")

with open(os.path.join(OUT_DIR, "final_text.txt"), "w") as f:
    for t in final_text:
        f.write(t + "\n")

print("\n✅ Saved final text to outputs/final_text.txt")