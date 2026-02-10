from paddleocr import PaddleOCR

print("🚀 USING PURE PADDLE OCR (PP-OCRv5 format)")

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True
)


def run_acc2_ocr(image_path):
    result = ocr.ocr(image_path)

    if not result or not isinstance(result, list):
        return "No text detected"

    page = result[0]

    # Your version stores recognized text here:
    rec_texts = page.get("rec_texts", [])

    if not rec_texts:
        return "No text extracted"

    # Clean and filter
    final_output = [
        text.strip()
        for text in rec_texts
        if isinstance(text, str) and len(text.strip()) > 2
    ]

    if not final_output:
        return "No text extracted"

    return "\n".join(final_output)