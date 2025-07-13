# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    portaudio19-dev \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional TTS dependencies
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/outputs /app/logs /app/models

# Set Python path
ENV PYTHONPATH=/app/src

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.ai_edu_qa_bot.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]