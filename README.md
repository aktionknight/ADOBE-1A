IN CASE DOCKER DOESNT WORK, run Predict_all.py . input directory for files input, output directory for output of extracted texts.

# PDF Heading and Title Extraction

## Approach

This solution uses a two-stage machine learning pipeline combined with **OCR fallback** for robust PDF title and heading extraction:

### Title Extraction
- Font-based Detection: Identifies titles by analyzing font size, position, and formatting characteristics
- OCR Fallback: When text blocks overlap or have complex layouts, switches to Tesseract OCR for accurate text extraction
- Multi-block Merging: Combines consecutive text blocks with similar formatting into complete titles
- Position Analysis: Prioritizes text in the top portion of the first page

### Heading Detection
- **Stage 1 - Heading vs Body**: RandomForest classifier distinguishes headings from body text using 30+ features
- **Stage 2 - Heading Levels**: Second RandomForest classifier assigns H1/H2/H3/H4 levels to detected headings
- **Hybrid Post-processing**: Combines ML predictions with heuristic rules for improved accuracy
- **Adaptive Processing**: Uses conservative detection for well-structured documents and permissive detection for complex layouts

### Feature Engineering
The system extracts 30+ features per text block including:
- **Typography**: Font size, bold/italic flags, font family, text case
- **Layout**: Position, alignment, line spacing, distance from top
- **Content**: Word count, text length, numbering patterns
- **Context**: Document position, heading density, page-level statistics

## Models and Libraries Used

### Core Libraries
- **PyMuPDF (fitz)**: PDF parsing and text extraction
- **scikit-learn**: Machine learning pipeline with RandomForest classifiers
- **numpy**: Numerical computations and array operations
- **joblib**: Model serialization and loading
- **Pillow (PIL)**: Image processing for OCR
- **pytesseract**: OCR engine with Japanese language support

### Machine Learning Models
- **RandomForest Classifier (Stage 1)**: 200 trees, balanced class weights for heading vs body classification
- **RandomForest Classifier (Stage 2)**: 200 trees, balanced class weights for H1/H2/H3/H4 level assignment
- **Feature Set**: 30+ engineered features per text block
- **Training Data**: Uses JSON annotations for supervised learning

### OCR Integration
- **Tesseract Engine**: Fallback OCR for complex text layouts
- **Japanese Support**: Multi-language OCR capabilities
- **Image Processing**: High-resolution rendering (2x scale) for better OCR accuracy

### Architecture
- **Offline Operation**: All models and dependencies included in container
- **AMD64 Compatible**: Optimized for x86_64 architecture
- **Memory Efficient**: <200MB total model size
- **Docker Containerized**: Isolated environment with frozen dependencies

This Docker container automatically processes PDF files to extract titles and headings, generating JSON output files.

## Features

- **AMD64 Architecture**: Compatible with x86_64 systems
- **Offline Operation**: No internet connectivity required
- **Multilingual Support**: Includes Japanese OCR support
- **Frozen Models**: Pre-trained models for fast inference
- **Automatic Processing**: Processes all PDFs in input directory

## Building the Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

## Running the Container

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

## Directory Structure

- **Input**: Place PDF files in the `input/` directory
- **Output**: JSON files will be generated in the `output/` directory

## Output Format

For each `filename.pdf`, a corresponding `filename.json` will be created with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Heading Text",
      "page": 0
    }
  ]
}
```

## Requirements

- Docker
- AMD64 architecture
- No GPU required
- Model size: < 200MB
- Offline operation

## Technical Details

- **Base Image**: Python 3.11-slim
- **OCR Engine**: Tesseract with Japanese support
- **ML Framework**: scikit-learn with RandomForest
- **PDF Processing**: PyMuPDF
- **Architecture**: linux/amd64
