# Medical Image Anonymization Pipeline - Production Docker Image
# Base: Python 3.11 on Debian (better compatibility with scientific packages)

FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, PaddleOCR, and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PaddlePaddle dependencies
    libgfortran5 \
    # EasyOCR dependencies
    libgeos-dev \
    # General utilities
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set PaddleOCR environment variables BEFORE any Python imports
# These MUST be set before paddleocr is imported
ENV FLAGS_use_mkldnn=0
ENV FLAGS_use_pir_api=0
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Additional environment variables for CPU-only mode
ENV CUDA_VISIBLE_DEVICES=""
ENV USE_GPU=0

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create directories for input/output (will be mounted as volumes)
RUN mkdir -p /app/input /app/output

# Pre-download models on build (optional but recommended)
# This reduces first-run time significantly
# Comment out if you want models downloaded on first run instead
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')" || true

# Set Python to run in unbuffered mode (better for Docker logs)
ENV PYTHONUNBUFFERED=1

# Use ENTRYPOINT to make container behave like an executable
# This allows: docker run anonymizer input.jpg output/
ENTRYPOINT ["python", "pipeline_run.py"]

# Default arguments (can be overridden)
CMD ["--help"]
