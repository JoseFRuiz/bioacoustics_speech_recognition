# Use Python 3.13 as base image (required for audioop-lts)
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies in a single layer
# FFmpeg is required for audio processing (pydub)
# --no-install-recommends avoids pulling unneeded GUI/X11/VA-API packages
# deb.debian.org's plain-HTTP endpoint has been returning 403 from its Fastly
# CDN; switching sources to HTTPS avoids it. Retries guard against remaining flakiness.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null; \
    echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries && \
    apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better Docker layer caching
COPY requirements.txt requirements-base.txt ./

# Install all Python dependencies in a single layer (faster rebuilds)
# Using pip cache mount for faster subsequent builds
# Note: Installation is split: base requirements first, then pyannote.audio separately
# torchaudio is installed with --no-deps to avoid version conflict with torch<2.6.0
# torchaudio 2.6.0+ works with torch 2.5.x for basic operations needed by pyannote.audio
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install "torch<2.6.0" && \
    pip install --no-deps torchaudio && \
    pip install -r requirements-base.txt && \
    pip install "pyannote.audio==4.0.3" && \
    pip install jupyter notebook

# Expose Jupyter notebook port
EXPOSE 8888

# Set default command to run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
