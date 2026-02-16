import cv2
import os
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

print("🔄 Loading TrOCR (Handwritten)...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"🚀 Using device: {device}")

# ---------------- HELPERS ----------------
def is_valid_crop(crop):
    if crop is None:
        return False
    h, w = crop.shape[:2]
    return h > 20 and w > 20


def preprocess_crop(crop):
    """
    Improve handwritten recognition accuracy
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Resize (important for transformer models)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold for handwriting
    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# ---------------- MAIN OCR ----------------
def main(image_path, save_debug=True):

    image_path = os.path.abspath(image_path)
    print("🧠 OCR reading:", image_path)

    if not os.path.exists(image_path):
        raise ValueError(f"❌ Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ OpenCV failed to read image: {image_path}")

    debug_img = img.copy()

    # ---- TEXT DETECTION ----
    results = ocr.predict(img)

    boxes = []
    for page in results:
        for poly in page["dt_polys"]:
            boxes.append(poly)

    if not boxes:
        return ["⚠️ No text detected"]

    # ---- SORT BOXES (Top to Bottom, Left to Right) ----
    boxes = sorted(
        boxes,
        key=lambda b: (
            np.min(np.array(b)[:, 1]),  # y-coordinate
            np.min(np.array(b)[:, 0]),  # x-coordinate
        ),
    )

    lines = []
    crop_id = 0

    # ---- RECOGNITION ----
    for poly in boxes:
        pts = np.array(poly, np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        # Add padding
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + pad)
        h = min(img.shape[0] - y, h + pad)

        crop = img[y : y + h, x : x + w]

        if not is_valid_crop(crop):
            continue

        # Save crop
        crop_path = os.path.join(CROP_DIR, f"crop_{crop_id}.jpg")
        cv2.imwrite(crop_path, crop)
        crop_id += 1

        # Preprocess
        processed_crop = preprocess_crop(crop)

        pil_img = Image.fromarray(processed_crop)

        pixel_values = processor(
            images=pil_img,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        if text:
            lines.append(text)

        # Draw bounding box
        cv2.rectangle(
            debug_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # ---- SAVE DEBUG IMAGE ----
    if save_debug:
        debug_path = os.path.join(OUTPUT_DIR, "debug_boxes.jpg")
        cv2.imwrite(debug_path, debug_img)
        print("🖼 Debug boxes saved:", debug_path)
        print("🟩 Crops saved in:", CROP_DIR)

    return lines


# ---------------- FLASK WRAPPER ----------------
def run_ocr(image_path):
    """
    Returns OCR result as single string
    """
    lines = main(image_path, save_debug=True)
    return "\n".join(lines)


# ---------------- RUN DIRECTLY ----------------
if __name__ == "__main__":
    result = run_ocr("uploads/test.jpg")
    print("\n📄 OCR OUTPUT:\n")
    print(result)