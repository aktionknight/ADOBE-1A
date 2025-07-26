import fitz
import pytesseract
from PIL import Image
import numpy as np
from text_block_extractor import extract_text_blocks

def predict_title_for_pdf(pdf_path):
    blocks = extract_text_blocks(pdf_path)
    page0_blocks = [b for b in blocks if b.get('page', 0) == 0]
    if not page0_blocks:
        return ""
    font_sizes = [b.get('font_size', 0) for b in page0_blocks]
    if not font_sizes:
        return ""
    max_font = max(font_sizes)
    # Only consider blocks with the largest font size
    candidates = [b for b in page0_blocks if abs(b.get('font_size', 0) - max_font) <= 1.5]
    # Sort by x (left to right), then y (top to bottom)
    candidates = sorted(candidates, key=lambda b: (b.get('origin', [0,0])[0], b.get('origin', [0,0])[1]))
    print(f"[DEBUG] {pdf_path}: {len(candidates)} candidate blocks for title")
    for b in candidates:
        print(f"[DEBUG] Block: text='{b.get('text','')}', x0={b.get('bbox',[0,0,0,0])[0]}, x1={b.get('bbox',[0,0,0,0])[2]}, y={b.get('origin',[0,0])[1]}")
    # Improved OCR trigger (x overlap only)
    ocr_triggered = False
    ocr_blocks = ()
    if len(candidates) >= 3:
        for i, b1 in enumerate(candidates):
            y1 = b1.get('origin', [0,0])[1]
            x1_0, x1_1 = b1.get('bbox', [0,0,0,0])[0], b1.get('bbox', [0,0,0,0])[2]
            for j, b2 in enumerate(candidates):
                if i == j:
                    continue
                y2 = b2.get('origin', [0,0])[1]
                x2_0, x2_1 = b2.get('bbox', [0,0,0,0])[0], b2.get('bbox', [0,0,0,0])[2]
                if abs(y1 - y2) <= 1.0:
                    # Compute x-overlap only
                    x_overlap = min(x1_1, x2_1) - max(x1_0, x2_0)
                    width1 = x1_1 - x1_0
                    width2 = x2_1 - x2_0
                    min_width = min(width1, width2)
                    if x_overlap > 0.5 * min_width:
                        ocr_triggered = True
                        ocr_blocks = (b1, b2)
                        print(f"[DEBUG] OCR TRIGGERED for {pdf_path} due to blocks: '{b1.get('text','')}' and '{b2.get('text','')}' with x-overlap {x_overlap}")
                        break
            if ocr_triggered:
                break
    if candidates and ocr_triggered:
        # Run OCR on the union of overlapping bboxes
        x0 = min(b['bbox'][0] for b in candidates)
        y0 = min(b['bbox'][1] for b in candidates)
        x1 = max(b['bbox'][2] for b in candidates)
        y1 = max(b['bbox'][3] for b in candidates)
        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = pytesseract.image_to_string(img, config='--psm 6').strip()
            print(f"[DEBUG] OCR result for {pdf_path}: '{ocr_text}'")
            if ocr_text:
                return ocr_text
        except Exception as e:
            print(f"[DEBUG] OCR error for {pdf_path}: {e}")
            pass
    # No OCR: use simple logic
    if len(candidates) == 1:
        title = candidates[0].get('text', '').strip()
        print(f"[DEBUG] Final single-block title for {pdf_path}: '{title}'")
        return title
    elif len(candidates) == 2:
        # Check for overlap
        b1, b2 = candidates
        x1_0, x1_1 = b1.get('bbox', [0,0,0,0])[0], b1.get('bbox', [0,0,0,0])[2]
        x2_0, x2_1 = b2.get('bbox', [0,0,0,0])[0], b2.get('bbox', [0,0,0,0])[2]
        if min(x1_1, x2_1) - max(x1_0, x2_0) > 0:
            # Overlap: just use the longer block
            title = max([b1, b2], key=lambda b: len(b.get('text', ''))).get('text', '').strip()
        else:
            # No overlap: concatenate left to right
            title = (b1.get('text', '') + ' ' + b2.get('text', '')).strip()
        print(f"[DEBUG] Final two-block title for {pdf_path}: '{title}'")
        return title
    else:
        # If more than 2 blocks but no OCR, use the longest block
        title = max(candidates, key=lambda b: len(b.get('text', ''))).get('text', '').strip()
        print(f"[DEBUG] Final multi-block (no OCR) title for {pdf_path}: '{title}'")
        return title 