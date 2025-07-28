import os
import json
import joblib
import subprocess
import sys
from predict_title import predict_title_for_pdf
from predict_headings import predict_headings_for_pdf

# Use local paths when running outside Docker
if os.path.exists('input'):
    INPUT_DIR = 'input'
    OUTPUT_DIR = 'output'
    STAGE1_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_stage1_heading_model.joblib')
    STAGE2_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_stage2_level_model.joblib')
else:
    # Docker paths
    INPUT_DIR = '/app/input'
    OUTPUT_DIR = '/app/output'
    STAGE1_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_stage1_heading_model.joblib')
    STAGE2_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_stage2_level_model.joblib')


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if models exist - they should be pre-trained during build
    if not os.path.exists(STAGE1_MODEL_PATH) or not os.path.exists(STAGE2_MODEL_PATH):
        print('Models not found. Training models...')
        try:
            subprocess.run([sys.executable, 'train_model.py'], check=True)
            print('Models trained successfully.')
        except subprocess.CalledProcessError as e:
            print(f'Error training models: {e}')
            print('Proceeding with prediction using default models...')
            return
    
    # Load pre-trained models
    print('Loading pre-trained models...')
    try:
        clf1 = joblib.load(STAGE1_MODEL_PATH)
        clf2 = joblib.load(STAGE2_MODEL_PATH)
        print('Models loaded successfully.')
    except Exception as e:
        print(f'Error loading models: {e}')
        return
    
    # Check if input directory exists and has PDFs
    if not os.path.exists(INPUT_DIR):
        print(f'Input directory {INPUT_DIR} does not exist.')
        return
    
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f'No PDF files found in {INPUT_DIR}')
        return
    
    print(f'Found {len(pdf_files)} PDF files to process: {pdf_files}')
    
    # Run heading and title prediction for all PDFs
    print(f'Running predictions for {len(pdf_files)} PDF files...')
    for fname in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, fname)
        base_name = fname[:-4]
        try:
            print(f'Processing {fname}...')
            # Predict title
            title = predict_title_for_pdf(pdf_path)
            # Predict headings using pre-trained models
            outline = predict_headings_for_pdf(pdf_path, clf1, clf2)
            # Save in train-style format
            out_path = os.path.join(OUTPUT_DIR, f'{base_name}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({'title': title, 'outline': outline}, f, ensure_ascii=False, indent=2)
            print(f"{fname}: title='{title}', headings={len(outline)}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if __name__ == '__main__':
    main() 
    