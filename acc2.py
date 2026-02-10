import cv2
import numpy as np
import torch
import re
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

TROCR_MODEL = "microsoft/trocr-large-handwritten"

print("🔄 Loading PaddleOCR...")
ocr = PaddleOCR(lang="en", use_textline_orientation=True)

print("🔄 Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


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
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


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

    dst = np.array([[0, 0], [maxW - 1, 0],
                    [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def box_center_y(box):
    return np.mean([p[1] for p in box])




def run_acc2_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ---- Horizontal projection ----
    horizontal_sum = np.sum(thresh, axis=1)

    lines = []
    start = None

    for i, val in enumerate(horizontal_sum):
        if val > 10 and start is None:
            start = i
        elif val <= 10 and start is not None:
            if i - start > 15:
                lines.append((start, i))
            start = None

    final_output = []

    for (y1, y2) in lines:
        line_crop = img[y1:y2, :]

        # Add padding
        pad = 20
        line_crop = cv2.copyMakeBorder(
            line_crop,
            pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

        # Resize to fixed height
        h, w = line_crop.shape[:2]
        target_h = 64
        scale = target_h / h
        new_w = int(w * scale)
        line_crop = cv2.resize(line_crop, (new_w, target_h))

        pil_img = Image.fromarray(line_crop).convert("RGB")

        pixel_values = processor(
            images=pil_img,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            ids = model.generate(
                pixel_values,
                max_length=256,
                num_beams=5
            )

        text = processor.batch_decode(
            ids,
            skip_special_tokens=True
        )[0].strip()

        if len(text) > 2:
            final_output.append(text)

    return "\n".join(final_output)