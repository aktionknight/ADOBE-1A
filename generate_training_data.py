import os
import json
from text_block_extractor import extract_text_blocks

def is_likely_heading(block):
    """Use heuristics to identify if a text block is likely a heading"""
    text = block.get('text', '').strip()
    if not text or len(text) < 3:
        return False
    
    # Check font size (headings are usually larger)
    font_size = block.get('font_size', 0)
    if font_size < 10:  # Too small to be a heading
        return False
    
    # Check if it's bold
    is_bold = block.get('is_bold', False)
    
    # Check if it's title case or all caps
    is_title_case = block.get('is_title_case', False)
    is_all_caps = block.get('is_all_caps', False)
    
    # Check word count (headings are usually short)
    word_count = block.get('word_count', 0)
    if word_count > 15:  # Too long to be a heading
        return False
    
    # Check if it starts with common heading patterns
    text_lower = text.lower()
    if any(text_lower.startswith(prefix) for prefix in [
        'chapter', 'section', 'part', 'appendix', 'introduction', 'conclusion',
        'abstract', 'summary', 'references', 'bibliography', 'index'
    ]):
        return True
    
    # Check if it ends with numbers (like "Chapter 1", "Section 2.1")
    if any(text.endswith(str(i)) for i in range(1, 20)):
        return True
    
    # Check if it's numbered (starts with numbers)
    words = text.split()
    if words and words[0].replace('.', '').isdigit():
        return True
    
    # Check if it's in title case or all caps
    if is_title_case or is_all_caps:
        return True
    
    # Check if it's bold and reasonably sized
    if is_bold and font_size >= 12:
        return True
    
    return False

def determine_heading_level(block, blocks):
    """Determine heading level based on font size and position"""
    font_size = block.get('font_size', 0)
    
    # Get all font sizes to determine relative sizing
    font_sizes = [b.get('font_size', 0) for b in blocks if b.get('font_size', 0) > 0]
    if not font_sizes:
        return 1
    
    max_font = max(font_sizes)
    min_font = min(font_sizes)
    
    # Determine level based on relative font size
    if font_size >= max_font * 0.9:  # Largest font
        return 1
    elif font_size >= max_font * 0.8:  # Large font
        return 2
    elif font_size >= max_font * 0.7:  # Medium font
        return 3
    else:  # Smaller font
        return 4

def generate_training_data():
    """Generate training data from PDF files in the train directory"""
    train_dir = 'train'
    output_dir = 'train_json'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for pdf_file in os.listdir(train_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(train_dir, pdf_file)
            json_file = pdf_file.replace('.pdf', '.json')
            json_path = os.path.join(output_dir, json_file)
            
            print(f"Processing {pdf_file}...")
            
            # Extract text blocks
            blocks = extract_text_blocks(pdf_path)
            
            # Identify headings
            headings = []
            for block in blocks:
                if is_likely_heading(block):
                    level = determine_heading_level(block, blocks)
                    headings.append({
                        'level': f'H{level}',
                        'text': block.get('text', '').strip(),
                        'page': block.get('page', 1)
                    })
            
            # Generate title (use the first large text block)
            title = "Untitled"
            for block in blocks:
                if block.get('font_size', 0) > 14 and len(block.get('text', '').strip()) > 5:
                    title = block.get('text', '').strip()
                    break
            
            # Create training data
            training_data = {
                'title': title,
                'outline': headings
            }
            
            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            print(f"  Generated {len(headings)} headings")
            print(f"  Title: {title}")
            print(f"  Saved to {json_path}")

if __name__ == "__main__":
    generate_training_data() 