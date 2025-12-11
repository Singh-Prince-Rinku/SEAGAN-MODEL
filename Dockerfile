# Dockerfile - CPU-only image for Render (uses official PyTorch CPU wheel index)
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install requirements (point pip to PyTorch CPU wheels)
RUN pip install --upgrade pip setuptools wheel \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt

# Copy project files
COPY . /app

# Create folders
RUN mkdir -p /app/data /app/checkpoints

# Expose port used by uvicorn
EXPOSE 8000

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
