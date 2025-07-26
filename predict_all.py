import subprocess
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from predict_title import predict_title_for_pdf
from predict_headings import predict_headings_for_pdf

def main():
    print("Extracting text blocks for all PDFs in input...")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'text_block_extractor.py')], check=True)
    
    # Model training is now frozen. To retrain, uncomment the next two lines.
    # print("Training model with maximum accuracy settings...")
    # subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'train_model.py')], check=True)
    
    print("Running predictions on test_input...")
    test_input_dir = os.path.join(os.path.dirname(__file__), 'test_input')
    test_output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(test_output_dir, exist_ok=True)
    
    for filename in os.listdir(test_input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(test_input_dir, filename)
            output_path = os.path.join(test_output_dir, filename.replace('.pdf', '.json'))
            
            # Predict title and headings
            title = predict_title_for_pdf(pdf_path)
            outline = predict_headings_for_pdf(pdf_path)
            
            # Save results
            result = {
                'title': title,
                'outline': outline
            }
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"{filename}: title='{title[:50]}{'...' if len(title) > 50 else ''}', headings={len(outline)}")

if __name__ == "__main__":
    main() 