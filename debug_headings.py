import os
from text_block_extractor import extract_text_blocks
from predict_headings import predict_headings_for_pdf
import joblib

def debug_file(filename):
    print(f"\n=== Debugging {filename} ===")
    
    # Load models
    clf1 = joblib.load('output/rf_stage1_heading_model.joblib')
    clf2 = joblib.load('output/rf_stage2_level_model.joblib')
    
    # Extract blocks
    pdf_path = os.path.join('input', filename)
    blocks = extract_text_blocks(pdf_path)
    print(f"Total blocks extracted: {len(blocks)}")
    
    # Show sample blocks
    print("\nSample blocks:")
    for i, block in enumerate(blocks[:10]):
        text = block.get('text', '')[:50]
        font_size = block.get('font_size', 0)
        is_bold = block.get('is_bold', False)
        is_title_case = block.get('is_title_case', False)
        print(f"{i}: '{text}'... Font: {font_size}, Bold: {is_bold}, TitleCase: {is_title_case}")
    
    # Run prediction
    outline = predict_headings_for_pdf(pdf_path, clf1, clf2)
    print(f"\nDetected headings: {len(outline)}")
    for heading in outline[:5]:
        print(f"  {heading['level']}: {heading['text']}")

if __name__ == "__main__":
    debug_file("file02.pdf") 