import joblib
from text_block_extractor import extract_text_blocks
import numpy as np

LEVEL_MAP_REV = {1: 'H1', 2: 'H2', 3: 'H3', 4: 'H4', 5: 'body'}
STAGE1_MODEL_PATH = 'output/rf_stage1_heading_model.joblib'
STAGE2_MODEL_PATH = 'output/rf_stage2_level_model.joblib'
HEADING_LABELS = ['H1', 'H2', 'H3', 'H4']
STAGE2_MAP_REV = {1: 'H1', 2: 'H2', 3: 'H3', 4: 'H4'}

def block_to_features(block):
    text_length = len(block.get('text', ''))
    is_upper = int(block.get('text', '').isupper())
    is_title = int(block.get('text', '').istitle())
    font_hash = hash(block.get('font', '')) % 10000
    flags_int = int(block.get('flags', 0)) if isinstance(block.get('flags', 0), (int, float)) else 0
    bbox = block.get('bbox', [0, 0, 0, 0])
    origin = block.get('origin', [0, 0])
    return [
        block.get('font_size', 0),
        text_length,
        is_upper,
        is_title,
        font_hash,
        flags_int,
        bbox[0], bbox[1], bbox[2], bbox[3],
        origin[0], origin[1],
        block.get('dist_from_top', 0.0),
        block.get('font_size_rank', 0),
        block.get('rel_font_size', 1.0),
        block.get('is_all_caps', 0),
        block.get('is_title_case', 0),
        block.get('line_gap_before', 0.0),
        block.get('ends_with_colon', 0),
        block.get('y_pct', 0.0),
        block.get('word_count', 0),
        block.get('starts_with_numbering', 0),
        int(block.get('alignment', 'other') == 'center'),
        int(block.get('alignment', 'other') == 'left'),
        int(block.get('alignment', 'other') == 'right'),
        int(block.get('alignment', 'other') == 'other'),
        block.get('font_is_unique', 0)
    ]

def predict_headings_for_pdf(pdf_path, clf1=None, clf2=None):
    if clf1 is None:
        clf1 = joblib.load(STAGE1_MODEL_PATH)
    if clf2 is None:
        clf2 = joblib.load(STAGE2_MODEL_PATH)
    blocks = extract_text_blocks(pdf_path)
    X = [block_to_features(b) for b in blocks]
    X = np.array(X)
    y1_pred = clf1.predict(X)
    headings_idx = np.where(y1_pred == 1)[0]
    outline = []
    if len(headings_idx) > 0:
        X_headings = X[headings_idx]
        y2_pred = clf2.predict(X_headings)
        for idx, level in zip(headings_idx, y2_pred):
            b = blocks[idx]
            lines = [line.strip() for line in b['text'].split('\n') if line.strip()]
            for line in lines:
                outline.append({
                    'level': STAGE2_MAP_REV.get(level, 'body'),
                    'text': line,
                    'page': b['page']
                })
    outline = sorted(outline, key=lambda h: (h['page'], blocks[0].get('origin', [0,0])[1], blocks[0].get('origin', [0,0])[0]) if blocks else (0,0,0))
    return outline 