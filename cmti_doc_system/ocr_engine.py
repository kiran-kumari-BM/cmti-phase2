from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang="en")


def run_ocr(image_path):
    try:
        result = ocr.ocr(image_path)

        if result is None:
            return ""

        extracted_lines = []

        # PaddleOCR returns list of pages
        if isinstance(result, list) and len(result) > 0:

            page = result[0]

            # New PaddleOCR format (dict)
            if isinstance(page, dict) and "rec_texts" in page:
                extracted_lines = page["rec_texts"]

            # Old format (list of boxes)
            elif isinstance(page, list):
                for line in page:
                    if len(line) >= 2:
                        text = line[1][0]
                        extracted_lines.append(text)

        return "\n".join(extracted_lines)

    except Exception as e:
        raise Exception(str(e))