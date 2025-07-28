import joblib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from text_block_extractor import extract_text_blocks
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

# Set fixed random seed for reproducibility
np.random.seed(42)

def create_default_models():
    """
    Create default models when no training data is available.
    These models will use basic heuristics for heading detection.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Create simple default models
    clf1 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    clf2 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Save models - use local path when running outside Docker
    if os.path.exists('output'):
        output_dir = 'output'
    else:
        output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(clf1, os.path.join(output_dir, 'rf_stage1_heading_model.joblib'))
    joblib.dump(clf2, os.path.join(output_dir, 'rf_stage2_level_model.joblib'))
    
    print("Default models created successfully.")

def block_to_features(block):
    text_length = len(block.get('text', ''))
    is_upper = int(block.get('text', '').isupper())
    is_title = int(block.get('text', '').istitle())
    font_hash = hash(block.get('font', '')) % 10000
    flags_int = int(block.get('flags', 0)) if isinstance(block.get('flags', 0), (int, float)) else 0
    bbox = block.get('bbox', [0, 0, 0, 0])
    origin = block.get('origin', [0, 0])
    features = [
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
        int(block.get('is_all_caps', False)),
        int(block.get('is_title_case', False)),
        int(block.get('is_bold', False)),
        block.get('line_gap_before', 0.0),
        block.get('ends_with_colon', 0),
        block.get('y_pct', 0.0),
        block.get('word_count', 0),
        block.get('starts_with_numbering', 0),
        # Alignment as numeric (0=left, 1=center, 2=right)
        block.get('alignment', 0),
        block.get('font_is_unique', 0),
        # Document-level features
        block.get('doc_position_rank', 0.0),
        int(block.get('is_first_heading_on_page', False)),
        block.get('heading_density', 0),
        block.get('page_heading_count', 0),
        block.get('doc_heading_count', 0)
    ]
    return features

def train_models():
    # Load training data from the current format (title and outline)
    train_dir = os.path.join(os.path.dirname(__file__), 'train')
    
    # Check if training data exists
    if not os.path.exists(train_dir):
        print(f"Training directory {train_dir} not found. Creating default models...")
        # Create simple default models
        create_default_models()
        return
    
    all_blocks = []
    all_labels = []
    
    # Load training data from JSON files
    for json_file in os.listdir(train_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(train_dir, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Get the corresponding PDF file
                pdf_name = json_file.replace('.json', '.pdf')
                pdf_path = os.path.join(train_dir, pdf_name)
                
                if os.path.exists(pdf_path):
                    print(f"Processing {pdf_name}...")
                    # Extract text blocks from PDF
                    blocks = extract_text_blocks(pdf_path)
                    print(f"  Extracted {len(blocks)} blocks")
                    
                    # Get the outline from training data
                    outline = data.get('outline', [])
                    print(f"  Training data has {len(outline)} headings")
                    
                    # Create a mapping of text to heading level with fuzzy matching
                    heading_map = {}
                    for item in outline:
                        if isinstance(item, dict) and 'text' in item and 'level' in item:
                            heading_text = item['text'].strip()
                            heading_map[heading_text] = item['level']
                    
                    # Label each block with fuzzy matching
                    matched_headings = 0
                    for block in blocks:
                        text = block.get('text', '').strip()
                        
                        # Try exact match first
                        if text in heading_map:
                            level = heading_map[text]
                            all_blocks.append(block)
                            all_labels.append(level)  # Use level directly, not f'H{level}'
                            matched_headings += 1
                            continue
                        
                        # Try fuzzy matching (normalize whitespace)
                        normalized_text = ' '.join(text.split())
                        if normalized_text in heading_map:
                            level = heading_map[normalized_text]
                            all_blocks.append(block)
                            all_labels.append(level)  # Use level directly, not f'H{level}'
                            matched_headings += 1
                            continue
                        
                        # Try partial matching for longer texts
                        matched = False
                        for heading_text, level in heading_map.items():
                            if heading_text in text or text in heading_text:
                                all_blocks.append(block)
                                all_labels.append(level)  # Use level directly, not f'H{level}'
                                matched_headings += 1
                                matched = True
                                break
                        
                        if not matched:
                            # This block is body text
                            all_blocks.append(block)
                            all_labels.append('body')
                    
                    print(f"  Matched {matched_headings} headings")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    if not all_blocks:
        print("No training data found!")
        return
    
    print(f"Loaded {len(all_blocks)} blocks for training")
    print(f"Labels distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    
    # Convert to features
    X = [block_to_features(b) for b in all_blocks]
    y = all_labels
    
    # Convert labels to numeric for Stage 1 (heading vs non-heading)
    y1 = []
    for label in y:
        if label in ['H1', 'H2', 'H3', 'H4']:
            y1.append(1)  # Heading
        else:
            y1.append(0)  # Body
    
    # Split data for Stage 1
    X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42, stratify=y1)
    
    # Stage 1: Heading vs Non-Heading
    print("Training Stage 1 model (Heading vs Non-Heading)...")
    clf1 = RandomForestClassifier(
        n_estimators=200,  # Increased for better accuracy
        max_depth=None,    # Allow full depth for maximum accuracy
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    clf1.fit(X_train, y1_train)
    
    # Evaluate Stage 1
    y1_pred = clf1.predict(X_test)
    print("Stage 1 Classification Report:")
    print(classification_report(y1_test, y1_pred))
    
    # Stage 2: Heading Levels (H1/H2/H3/H4)
    print("\nTraining Stage 2 model (Heading Levels)...")
    
    # Get only heading blocks for Stage 2
    heading_indices = [i for i, label in enumerate(y) if label in ['H1', 'H2', 'H3', 'H4']]
    X_headings = [X[i] for i in heading_indices]
    y_headings = [y[i] for i in heading_indices]
    
    # Convert heading levels to numeric (H1=1, H2=2, H3=3, H4=4)
    y_headings_numeric = []
    for label in y_headings:
        if label.startswith('H'):
            level = int(label[1])
            y_headings_numeric.append(level)
        else:
            y_headings_numeric.append(1)  # Default to H1
    
    if len(X_headings) > 0:
        # Split heading data
        X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
            X_headings, y_headings_numeric, test_size=0.2, random_state=42, stratify=y_headings_numeric
        )
        
        clf2 = RandomForestClassifier(
            n_estimators=200,  # Increased for better accuracy
            max_depth=None,    # Allow full depth for maximum accuracy
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        clf2.fit(X_h_train, y_h_train)
        
        # Evaluate Stage 2
        y2_pred = clf2.predict(X_h_test)
        print("Stage 2 Classification Report:")
        print(classification_report(y_h_test, y2_pred))
    else:
        print("No heading blocks found for Stage 2 training. Creating default model.")
        # Create a default model for Stage 2
        clf2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        # Fit with dummy data to ensure the model is valid
        dummy_X = [[0] * 30]  # 30 features
        dummy_y = [1]  # Default to H1
        clf2.fit(dummy_X, dummy_y)
    
    # Save models - use local path when running outside Docker
    if os.path.exists('output'):
        output_dir = 'output'
    else:
        output_dir = '/app/output'
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(clf1, os.path.join(output_dir, 'rf_stage1_heading_model.joblib'))
    joblib.dump(clf2, os.path.join(output_dir, 'rf_stage2_level_model.joblib'))
    
    print(f"\nModels saved to {output_dir}")
    print("Stage 1 model saved to rf_stage1_heading_model.joblib")
    print("Stage 2 model saved to rf_stage2_level_model.joblib")

if __name__ == "__main__":
    train_models() 