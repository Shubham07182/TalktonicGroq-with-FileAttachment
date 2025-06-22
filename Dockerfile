# Use a base image with Python
FROM python:3.10-slim

# Install system dependencies (Tesseract + poppler for pdfplumber + others)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Streamlit
ENV PYTHONUNBUFFERED=1

# Run the app
CMD ["streamlit", "run", "LLM.py", "--server.port=8501", "--server.enableCORS=false"]
