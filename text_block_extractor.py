import fitz
import os
import json
import numpy as np

def extract_text_blocks(pdf_path, save_to_output=True):
    """
    Extracts text blocks and their features from all pages of a PDF using PyMuPDF.
    Returns a list of dicts: {page, text, font_size, bbox, font, flags, origin, dist_from_top, font_size_rank, rel_font_size, is_all_caps, is_title_case, line_gap_before, ends_with_colon, y_pct, word_count, starts_with_numbering, alignment, font_is_unique}
    If save_to_output is True, saves the extracted blocks as a JSON file in the test folder.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        page_blocks = []
        prev_y = None
        font_counter = {}
        for b in page.get_text("dict")['blocks']:
            if 'lines' not in b:
                continue
            for line in b['lines']:
                for span in line['spans']:
                    text = span['text'].strip()
                    y = span['bbox'][1]
                    line_gap_before = 0.0
                    if prev_y is not None:
                        line_gap_before = round(y - prev_y, 2)
                    prev_y = y
                    font = span['font']
                    font_counter[font] = font_counter.get(font, 0) + 1
                    block = {
                        'page': page_num,
                        'text': text,
                        'font_size': round(span['size'], 1),
                        'bbox': span['bbox'],
                        'font': font,
                        'flags': span['flags'],
                        'origin': (span['bbox'][0], span['bbox'][1]),
                        'dist_from_top': span['bbox'][1] / page_height if page_height else 0.0,
                        'line_gap_before': line_gap_before,
                        'is_all_caps': int(text.isupper()),
                        'is_title_case': int(text.istitle()),
                        'ends_with_colon': int(text.endswith(':')),
                        'y_pct': span['bbox'][1] / page_height if page_height else 0.0,
                        'word_count': len(text.split()),
                        'starts_with_numbering': int(bool(text.split() and (text.split()[0].rstrip('.').isdigit() or text.split()[0][:-1].isdigit()))),
                    }
                    # Alignment: centered, left, right (approximate)
                    x0, x1 = span['bbox'][0], span['bbox'][2]
                    center = (x0 + x1) / 2
                    if abs(center - page_width / 2) < 0.1 * page_width:
                        block['alignment'] = 'center'
                    elif x0 < 0.1 * page_width:
                        block['alignment'] = 'left'
                    elif x1 > 0.9 * page_width:
                        block['alignment'] = 'right'
                    else:
                        block['alignment'] = 'other'
                    page_blocks.append(block)
        # font_size_rank and rel_font_size for this page
        font_sizes = [b['font_size'] for b in page_blocks]
        median_font_size = float(np.median(font_sizes)) if font_sizes else 1.0
        sorted_blocks = sorted(page_blocks, key=lambda b: -b['font_size'])
        for i, b in enumerate(sorted_blocks):
            b['font_size_rank'] = (i+1) / len(page_blocks) if page_blocks else 1.0
            b['rel_font_size'] = b['font_size'] / median_font_size if median_font_size else 1.0
        # font_is_unique for this page
        for b in page_blocks:
            b['font_is_unique'] = int(font_counter.get(b['font'], 0) == 1)
        blocks.extend(page_blocks)
    if save_to_output:
        os.makedirs('test', exist_ok=True)
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join('test', f'{base}_blocks.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(blocks, f, ensure_ascii=False, indent=2)
    return blocks

if __name__ == '__main__':
    import glob
    input_dir = 'input'
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    for pdf_path in pdf_files:
        print(f"Extracting blocks from {pdf_path}...")
        extract_text_blocks(pdf_path, save_to_output=True)
    print(f"Extraction complete. JSON files saved to 'test' folder.") 