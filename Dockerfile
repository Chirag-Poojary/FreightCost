# ==============================================================================
# FreightCost AI - Production Container
# Includes Linux Tesseract OCR binary, Python 3.11, XGBoost & FastAPI
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies including Tesseract OCR and image libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application, extraction scripts, and trained model artifacts
COPY backend/ backend/
COPY extraction_scripts/ extraction_scripts/
COPY code/ code/
COPY phase_a_output/ phase_a_output/
COPY frontend/ frontend/

# Expose default port
ENV PORT=8000
EXPOSE 8000

# Start FastAPI via uvicorn
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
