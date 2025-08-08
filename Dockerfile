# Dockerfile for Plant Disease Classifier FastAPI Inference API
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install pipdeptree for debugging dependencies
RUN pip install pipdeptree

# Debugging step: Generate dependency tree and save to a file
RUN pipdeptree > /app/dependency-tree.txt

# Debugging step: List all installed packages
RUN pip list > /app/installed-packages.txt

# Ensure multipart is removed
RUN pip uninstall -y multipart

# Copy model and app code
COPY models/vit-finetuned ./models/vit-finetuned
COPY src/inference/app.py ./src/inference/app.py

# Expose port
EXPOSE 8000

# Start FastAPI app with Uvicorn (use PORT env variable set by Render)
CMD uvicorn src.inference.app:app --host 0.0.0.0 --port $PORT
