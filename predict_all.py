import os
import json
import joblib
import subprocess
from predict_title import predict_title_for_pdf
from predict_headings import predict_headings_for_pdf
from text_block_extractor import extract_text_blocks

INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
MODEL_PATH = 'output/rf_heading_model.joblib'


def main():
    # 1. Run text block extraction for all PDFs
    print('Extracting text blocks for all PDFs...')
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(INPUT_DIR, fname)
        extract_text_blocks(pdf_path, save_to_output=True)
    # 2. Run train_model.py to retrain the model
    print('Training model...')
    subprocess.run(['python', 'train_model.py'], check=True)
    # 3. Run heading and title prediction for all PDFs
    print('Running predictions...')
    clf1 = None
    clf2 = None
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(INPUT_DIR, fname)
        base_name = fname[:-4]
        # Predict title
        title = predict_title_for_pdf(pdf_path)
        # Predict headings
        outline = predict_headings_for_pdf(pdf_path, clf1, clf2)
        # Save in train-style format
        out_path = os.path.join(OUTPUT_DIR, f'{base_name}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'title': title, 'outline': outline}, f, ensure_ascii=False, indent=2)
        print(f"{fname}: title='{title}', headings={len(outline)}")

if __name__ == '__main__':
    main() 