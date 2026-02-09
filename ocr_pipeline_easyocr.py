import cv2
import os
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np

# ---------------- CONFIG ----------------
IMAGE_PATH = "test.jpg"   # image must be in cmtii/
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_name = "microsoft/trocr-large-handwritten"

# ---------------- LOAD MODELS ----------------
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False)

print("Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ---------------- LOAD IMAGE ----------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError("❌ Image not found! Check IMAGE_PATH")

# ---------------- DETECT BOXES ----------------
results = reader.readtext(IMAGE_PATH)
print("Detected boxes:", len(results))

# ---------------- SORT BOXES (TOP→BOTTOM, LEFT→RIGHT) ----------------
def box_center(bbox):
    x = sum([p[0] for p in bbox]) / 4
    y = sum([p[1] for p in bbox]) / 4
    return x, y

results = sorted(results, key=lambda r: (box_center(r[0])[1], box_center(r[0])[0]))

# ---------------- DRAW BOXES ----------------
boxed = img.copy()

for (bbox, text, conf) in results:
    pts = np.array([[int(x), int(y)] for x, y in bbox], np.int32)
    cv2.polylines(boxed, [pts], True, (0, 255, 0), 2)

cv2.imwrite(os.path.join(OUTPUT_DIR, "boxes.jpg"), boxed)
print("✅ Saved boxes image to outputs/boxes.jpg")

# ---------------- OCR EACH BOX WITH TrOCR ----------------
final_text = []

for (bbox, _, _) in results:
    x_coords = [int(p[0]) for p in bbox]
    y_coords = [int(p[1]) for p in bbox]

    x1, x2 = max(0, min(x_coords)), min(img.shape[1], max(x_coords))
    y1, y2 = max(0, min(y_coords)), min(img.shape[0], max(y_coords))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        ids = model.generate(
            pixel_values,
            max_length=32,
            num_beams=5,
            early_stopping=True
        )

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    text = text.strip().replace(".", "").replace(",", "")

    if len(text) > 1:
        final_text.append(text)

# ---------------- FORMAT FINAL TEXT INTO LINES ----------------
lines = []
current_line = []
prev_y = None

for (bbox, _, _), word in zip(results, final_text):
    _, cy = box_center(bbox)

    if prev_y is None or abs(cy - prev_y) < 40:
        current_line.append(word)
    else:
        lines.append(" ".join(current_line))
        current_line = [word]

    prev_y = cy

if current_line:
    lines.append(" ".join(current_line))

print("\n========== FINAL OCR ==========\n")
for line in lines:
    print(line)