import cv2
import os
import torch
import numpy as np
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import fitz  # PyMuPDF for PDF support

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CROP_DIR = os.path.join(OUTPUT_DIR, "crops")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# ---------------- LOAD MODELS ----------------
print("🔄 Loading PaddleOCR...")
# Adjusted for better LINE detection (not paragraph)
ocr = PaddleOCR(
    lang="en",
    det_db_box_thresh=0.3,      # Lower threshold for more boxes
    det_db_unclip_ratio=1.5,    # Less aggressive box expansion
    use_angle_cls=False         # Faster, good for horizontal text
)
print("✅ PaddleOCR loaded!")

print("🔄 Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
print(f"✅ TrOCR loaded on {device}!")

# ---------------- HELPERS ----------------
def is_valid_crop(crop):
    if crop is None:
        return False
    h, w = crop.shape[:2]
    return h > 15 and w > 15

def split_box_into_lines(img, box, max_height=80):
    """
    Split a large box (paragraph) into individual lines
    """
    pts = np.array(box, np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    
    # If box is already small (single line), return as-is
    if h <= max_height:
        return [box]
    
    # Otherwise, split into multiple lines
    crop = img[y:y+h, x:x+w]
    
    # Convert to grayscale for line detection
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection to find text lines
    horizontal_projection = np.sum(binary, axis=1)
    
    # Find line breaks (where projection is low)
    threshold = np.max(horizontal_projection) * 0.1
    in_line = False
    line_starts = []
    line_ends = []
    
    for i, val in enumerate(horizontal_projection):
        if val > threshold and not in_line:
            line_starts.append(i)
            in_line = True
        elif val <= threshold and in_line:
            line_ends.append(i)
            in_line = False
    
    # Handle last line
    if in_line:
        line_ends.append(len(horizontal_projection))
    
    # Create new boxes for each line
    new_boxes = []
    for start, end in zip(line_starts, line_ends):
        if end - start > 10:  # Min line height
            # Create box coordinates in original image space
            new_box = [
                [x, y + start],
                [x + w, y + start],
                [x + w, y + end],
                [x, y + end]
            ]
            new_boxes.append(new_box)
    
    return new_boxes if new_boxes else [box]

def pdf_to_images(pdf_path, dpi=150):
    """Convert PDF pages to images"""
    print(f"📄 Converting PDF: {os.path.basename(pdf_path)}")
    images = []
    
    pdf_doc = fitz.open(pdf_path)
    total_pages = len(pdf_doc)
    print(f"   Total pages: {total_pages}")
    
    for page_num in range(total_pages):
        page = pdf_doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        if pix.n == 4:
            img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        else:
            img = img_data
        
        images.append(img)
        print(f"   ✅ Page {page_num + 1}/{total_pages}")
    
    pdf_doc.close()
    return images

# ---------------- MAIN OCR ----------------
def process_image(img, page_num=0, save_debug=True):
    """
    Improved line-level detection + simple reading
    """
    print(f"🧠 OCR processing page {page_num + 1}...")
    
    debug_img = img.copy()
    
    # PaddleOCR detection
    results = ocr.predict(img)
    
    # Extract boxes
    boxes = []
    for page in results:
        if isinstance(page, dict) and "dt_polys" in page:
            boxes = page["dt_polys"]
            break
    
    if not boxes:
        print(f"   ⚠️ No text detected on page {page_num + 1}")
        return []
    
    print(f"   Initial boxes: {len(boxes)}")
    
    # Split large boxes into lines
    all_line_boxes = []
    for box in boxes:
        line_boxes = split_box_into_lines(img, box, max_height=80)
        all_line_boxes.extend(line_boxes)
    
    print(f"   After line splitting: {len(all_line_boxes)} boxes")
    
    lines = []
    crop_id = 0
    
    # Sort boxes top-to-bottom
    sorted_boxes = sorted(all_line_boxes, key=lambda box: np.mean([pt[1] for pt in box]))
    
    for poly in sorted_boxes:
        pts = np.array(poly, np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        
        # Small padding
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        crop = img[y_start:y_end, x_start:x_end]
        
        if not is_valid_crop(crop):
            continue
        
        # Save crop
        crop_path = os.path.join(CROP_DIR, f"page{page_num}_crop_{crop_id}.jpg")
        cv2.imwrite(crop_path, crop)
        crop_id += 1
        
        # Simple conversion - NO preprocessing
        pil_img = Image.fromarray(crop).convert("RGB")
        
        # TrOCR inference
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=128)
        
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        
        if text:
            lines.append(text)
        
        # Draw box (different color for split boxes)
        color = (0, 255, 0) if len(all_line_boxes) > len(boxes) else (255, 0, 0)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
    
    # Save debug
    if save_debug:
        debug_path = os.path.join(OUTPUT_DIR, f"debug_page{page_num}_boxes.jpg")
        cv2.imwrite(debug_path, debug_img)
        print(f"   🖼 Debug saved: {debug_path}")
    
    print(f"   ✅ Extracted {len(lines)} lines")
    return lines

def main(file_path, save_debug=True):
    """Main OCR function"""
    file_path = os.path.abspath(file_path)
    print(f"\n{'='*60}")
    print(f"🚀 Starting OCR with Line Detection: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        raise ValueError(f"❌ File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    all_lines = []
    
    # Handle PDFs
    if file_ext == ".pdf":
        images = pdf_to_images(file_path)
        
        for page_num, img in enumerate(images):
            lines = process_image(img, page_num, save_debug)
            
            if lines:
                all_lines.append(f"\n--- Page {page_num + 1} ---\n")
                all_lines.extend(lines)
    
    # Handle images
    elif file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        img = cv2.imread(file_path)
        
        if img is None:
            raise ValueError(f"❌ OpenCV failed to read image: {file_path}")
        
        lines = process_image(img, 0, save_debug)
        all_lines.extend(lines)
    
    else:
        raise ValueError(f"❌ Unsupported file type: {file_ext}")
    
    if not all_lines:
        return ["⚠️ No text detected"]
    
    print(f"\n{'='*60}")
    print(f"✅ OCR Complete!")
    print(f"   Total lines: {len([l for l in all_lines if not l.startswith('---')])}")
    print(f"{'='*60}")
    print(f"🟩 Crops saved in: {CROP_DIR}\n")
    
    return all_lines

# ---------------- BACKEND WRAPPER ----------------
def run_ocr(file_path):
    """Flask wrapper"""
    try:
        lines = main(file_path, save_debug=True)
        return "\n".join(lines)
    except Exception as e:
        import traceback
        error_msg = f"OCR Error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return error_msg