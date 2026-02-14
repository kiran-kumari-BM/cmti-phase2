import cv2
import os
import re
import torch
import numpy as np
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# ---------------- LOAD MODELS (CACHED) ----------------
print("🔄 Loading PaddleOCR...")
ocr = PaddleOCR(lang="en", use_textline_orientation=True)

print("🔄 Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# ---------------- HELPERS ----------------
def is_valid_crop(crop):
    if crop is None:
        return False
    h, w = crop.shape[:2]
    return h > 15 and w > 15

# ---------------- MAIN OCR ----------------
def main(image_path, save_debug=True):
    # ✅ Ensure absolute path
    image_path = os.path.abspath(image_path)
    print("🧠 OCR reading:", image_path)

    if not os.path.exists(image_path):
        raise ValueError(f"❌ Image not found on disk: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ OpenCV failed to read image: {image_path}")

    debug_img = img.copy()
    results = ocr.predict(img)

    boxes = []
    for page in results:
        for poly in page["dt_polys"]:
            boxes.append(poly)

    if not boxes:
        return ["⚠️ No text detected"]

    lines = []
    crop_id = 0

    for poly in boxes:
        pts = np.array(poly, np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        crop = img[y:y+h, x:x+w]
        if not is_valid_crop(crop):
            continue

        crop_path = os.path.join(CROP_DIR, f"crop_{crop_id}.jpg")
        cv2.imwrite(crop_path, crop)
        crop_id += 1

        pil_img = Image.fromarray(crop).convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=128)

        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            lines.append(text)

        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if save_debug:
        debug_path = os.path.join(OUTPUT_DIR, "debug_boxes.jpg")
        cv2.imwrite(debug_path, debug_img)
        print("🖼 Debug boxes saved:", debug_path)
        print("🟩 Crops saved in:", CROP_DIR)

    return lines

# ---------------- BACKEND WRAPPER ----------------
def run_ocr(image_path):
    """
    Flask wrapper
    Returns OCR text as a single string
    """
    lines = main(image_path, save_debug=True)
    return "\n".join(lines)

if __name__ == "__main__":
    print(run_ocr("uploads/test.jpg"))