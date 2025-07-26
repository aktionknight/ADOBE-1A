import os
import json
import numpy as np
from text_block_extractor import extract_text_blocks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report
import joblib

LEVEL_MAP = {
    'H1': 1,
    'H2': 2,
    'H3': 3,
    'H4': 4,
    'body': 5
}
LEVEL_MAP_REV = {v: k for k, v in LEVEL_MAP.items()}

TRAIN_DIR = 'train'
INPUT_DIR = 'input'
MODEL_PATH = 'output/rf_heading_model.joblib'

HEADING_LABELS = ['H1', 'H2', 'H3', 'H4']
HEADING_LEVELS = {k: v for k, v in LEVEL_MAP.items() if k in HEADING_LABELS}

# Stage 1: Heading vs. Non-Heading
STAGE1_MAP = {'heading': 1, 'body': 0}
STAGE1_MAP_REV = {v: k for k, v in STAGE1_MAP.items()}

# Stage 2: H1/H2/H3/H4 (for heading blocks only)
STAGE2_MAP = {k: i+1 for i, k in enumerate(HEADING_LABELS)}
STAGE2_MAP_REV = {v: k for k, v in STAGE2_MAP.items()}

STAGE1_MODEL_PATH = 'output/rf_stage1_heading_model.joblib'
STAGE2_MODEL_PATH = 'output/rf_stage2_level_model.joblib'


def find_label(block, outline):
    def norm(s):
        return ''.join(s.lower().split())
    for item in outline:
        if block['page'] == item['page'] and norm(block['text']) == norm(item['text']) and item['level'] in ['H1', 'H2', 'H3', 'H4']:
            return item['level']
    return 'body'

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
        # Alignment as one-hot
        int(block.get('alignment', 'other') == 'center'),
        int(block.get('alignment', 'other') == 'left'),
        int(block.get('alignment', 'other') == 'right'),
        int(block.get('alignment', 'other') == 'other'),
        block.get('font_is_unique', 0)
    ]

def train_model():
    X_list, y1_list, y2_list, file_names = [], [], [], []
    for fname in os.listdir(TRAIN_DIR):
        if not fname.endswith('.json'):
            continue
        base = fname[:-5]
        pdf_path = os.path.join(INPUT_DIR, base + '.pdf')
        label_path = os.path.join(TRAIN_DIR, fname)
        if not os.path.exists(pdf_path):
            continue
        with open(label_path, encoding='utf-8') as f:
            label_data = json.load(f)
        outline = label_data.get('outline', [])
        blocks = extract_text_blocks(pdf_path)
        for block in blocks:
            label = find_label(block, outline)
            X_list.append(block_to_features(block))
            # Stage 1 label: heading vs. non-heading
            y1_list.append(1 if label in HEADING_LABELS else 0)
            # Stage 2 label: heading level (only for headings)
            y2_list.append(STAGE2_MAP.get(label, 0))
            file_names.append(base)
    X = np.array(X_list)
    y1 = np.array(y1_list)
    y2 = np.array(y2_list)
    file_names = np.array(file_names)
    unique_files = sorted(set(file_names))
    # Stage 1: Heading vs. Non-Heading
    loo = LeaveOneOut()
    y1_true_all, y1_pred_all = [], []
    y2_true_all, y2_pred_all = [], []
    for train_idx, test_idx in loo.split(unique_files):
        test_file = unique_files[test_idx[0]]
        train_mask = file_names != test_file
        test_mask = file_names == test_file
        X_train, y1_train, y2_train = X[train_mask], y1[train_mask], y2[train_mask]
        X_test, y1_test, y2_test = X[test_mask], y1[test_mask], y2[test_mask]
        # Stage 1 model
        clf1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
        clf1.fit(X_train, y1_train)
        y1_pred = clf1.predict(X_test)
        y1_true_all.extend(y1_test)
        y1_pred_all.extend(y1_pred)
        # Stage 2 model (only for heading blocks)
        idx_headings = np.where(y1_pred == 1)[0]
        if len(idx_headings) > 0:
            clf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
            clf2.fit(X_train[y1_train == 1], y2_train[y1_train == 1])
            y2_pred = clf2.predict(X_test[idx_headings])
            y2_true = y2_test[idx_headings]
            y2_true_all.extend(y2_true)
            y2_pred_all.extend(y2_pred)
    print("\n=== Stage 1: Heading vs. Non-Heading (LOO-CV) ===")
    print(classification_report(y1_true_all, y1_pred_all, labels=[0,1], target_names=['body','heading']))
    print("\n=== Stage 2: Heading Level (LOO-CV, only for predicted headings) ===")
    if y2_true_all:
        print(classification_report(y2_true_all, y2_pred_all, labels=list(STAGE2_MAP.values()), target_names=HEADING_LABELS))
    else:
        print("No headings predicted in LOO-CV.")
    # Train final models on all data
    clf1 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    clf1.fit(X, y1)
    joblib.dump(clf1, STAGE1_MODEL_PATH)
    print(f"Stage 1 model saved to {STAGE1_MODEL_PATH}")
    if np.any(y1 == 1):
        clf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
        clf2.fit(X[y1 == 1], y2[y1 == 1])
        joblib.dump(clf2, STAGE2_MODEL_PATH)
        print(f"Stage 2 model saved to {STAGE2_MODEL_PATH}")
    else:
        print("No heading blocks found for Stage 2 model.")

if __name__ == '__main__':
    train_model() 