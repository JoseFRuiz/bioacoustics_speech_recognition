# Use Python 3.13 as base image (required for audioop-lts)
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# FFmpeg is required for audio processing (pydub)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Jupyter for notebook support
RUN pip install --no-cache-dir jupyter notebook

# Copy project files
COPY . .

# Expose Jupyter notebook port
EXPOSE 8888

# Set default command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
