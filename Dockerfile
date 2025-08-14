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

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install pipdeptree for debugging dependencies
RUN pip install pipdeptree

# Debugging step: Generate dependency tree and save to a file
RUN pipdeptree > /app/dependency-tree.txt

# Debugging step: List all installed packages
RUN pip list > /app/installed-packages.txt

# Debug: Output Python version and environment variables
RUN python --version > /app/python-version.txt
RUN env > /app/env.txt

# Ensure multipart is removed
RUN pip uninstall -y multipart

# Copy model and app code
COPY models/vit-finetuned ./models/vit-finetuned

# Copy FastAPI and Streamlit apps
COPY ./app.py ./app.py
#COPY ./src/inference/app_fastapi.py ./src/inference/app_fastapi.py

# Copy assets
COPY ./images/plant-disease-logo.png ./images/plant-disease-logo.png
# fastapi app needs static files to serve
#COPY ./static ./src/inference/static
COPY ./README.md ./README.md

# Copy other source code
COPY ./src ./src

# If model.py is in the root
#COPY ./model.py ./model.py

# Set Streamlit config/cache directory to a writable location
ENV HOME=/tmp
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit

# Expose ports for Streamlit (7860) and FastAPI (8000)
EXPOSE 7860
#EXPOSE 8000

# Default to Streamlit; override in docker run if you want FastAPI
# to run FastAPI, we need to change the Dockerfile CMD and ensure the FastAPI app is set up to serve index.html and /predict/.
# Debug: On container start, print environment info before running the app
CMD python -c "import sys, os; print('Python:', sys.version); print('Env:', dict(os.environ));" && streamlit run app.py --server.port=7860 --server.address=0.0.0.0

# Debug: List files in /app after copying everything
RUN ls -lR /app > /app/files-list.txt

# Debug: On container start, print environment info and list files before running the app
CMD python -c "import sys, os; print('Python:', sys.version); print('Env:', dict(os.environ)); os.system('ls -lR /app');" && streamlit run app.py --server.port=7860 --server.address=0.0.0.0

# To run FastAPI, uncomment below and comment out the Streamlit CMD above
#CMD python -c "import sys, os; print('Python:', sys.version); print('Env:', dict(os.environ)); os.system('ls -lR /app');" && uvicorn src.inference.app_fastapi:app --host 0.0.0.0 --port 7860