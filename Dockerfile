FROM --platform=linux/amd64 python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-jpn \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Copy training data and input files
COPY train/ ./train/
COPY input/ ./input/

# Create output directory
RUN mkdir -p output

# Pre-train models during build to ensure consistency
RUN python train_model.py

# Set the entrypoint to run the prediction script
ENTRYPOINT ["python", "predict_all.py"] 