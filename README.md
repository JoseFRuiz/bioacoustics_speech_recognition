# bioacoustics_speech_recognition

## Environment Setup (Docker)

### Prerequisites

- **Docker**: Install Docker Desktop or Docker Engine from [https://www.docker.com/get-started](https://www.docker.com/get-started)
- **Docker Compose** (usually included with Docker Desktop): For easier container management

### Setup Steps

**1) Grant permissions for pyannote models**

Before using the voice activity detection features, you need to accept the terms and conditions for the pyannote models:

- Visit [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- Log in to your Hugging Face account
- Accept the access conditions for the model
- Create an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) if you haven't already
- Create a `.env` file in the repository root folder and add your Hugging Face token:

  ```bash
  # Create .env file in the repo root
  echo HF_TOKEN=your_token_here > .env
  ```
  
  Or manually create a `.env` file with the following content:
  ```
  HF_TOKEN=your_token_here
  ```

**2) Build and run the Docker container**

Using Docker Compose (recommended):

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### Managing the Container

**Stop the container:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

**Execute commands in the running container:**
```bash
docker-compose exec bioacoustics bash
```

**Rebuild the Docker image:**

Rebuilding the Docker image is necessary when you:
- Update `Dockerfile` or `docker-compose.yml`
- Add or modify system dependencies
- Change Python version or base image
- Add new system packages or tools (e.g., Poetry, new build tools)
- Update `requirements.txt` and want to ensure clean installation

**Using Docker Compose (recommended):**
```bash
# Stop any running containers first
docker-compose down

# Rebuild and start the container
docker-compose up --build

# Or rebuild without starting
docker-compose build

# Or rebuild without cache (slower but ensures clean build)
docker-compose build --no-cache
```

**Note:** After rebuilding, you may need to restart your container if it was already running. Use `docker-compose down` followed by `docker-compose up` to ensure you're using the newly built image.

### Faster Build Tips

To speed up Docker container builds:

**1. Use BuildKit (enabled by default in Docker Desktop):**
```bash
# Build with BuildKit for cache mounts and parallel builds
DOCKER_BUILDKIT=1 docker-compose build
```

**2. Leverage layer caching:**
- The Dockerfile is optimized to install dependencies before copying code
- Only rebuilds dependencies when `requirements.txt` changes
- Code changes don't trigger dependency reinstallation

**3. Use build cache:**
```bash
# Build using cache (default behavior)
docker-compose build

# Skip cache only when needed (much slower)
docker-compose build --no-cache
```

**4. Build specific services:**
```bash
# Only build the bioacoustics service
docker-compose build bioacoustics
```

**5. Parallel dependency installation:**
The Dockerfile uses pip cache mounts to speed up subsequent builds. The first build downloads packages, but subsequent builds reuse cached packages.

**6. Exclude unnecessary files:**
The `.dockerignore` file excludes output files, data, and other files that don't need to be in the image, reducing build context size.

### Running Python Scripts

You can run Python scripts (like `TransformerDetection.py`) in the container using several methods:

**Option 1: Execute script in running container (recommended)**

If the container is already running (via `docker-compose up`), execute the script directly:

```bash
docker-compose exec bioacoustics python TransformerDetection.py
```

**Option 2: Run as one-off command**

This starts a temporary container, runs the script, and exits automatically:

```bash
docker-compose run --rm bioacoustics python TransformerDetection.py
```

**Option 3: Interactive shell**

Enter the container and run commands interactively:

```bash
# Enter the container
docker-compose exec bioacoustics bash

# Then inside the container, run:
python TransformerDetection.py

# Exit when done
exit
```

**Option 4: Using Docker directly (without docker-compose)**

```bash
docker run --rm -v "${PWD}:/app" -v "${PWD}/data:/app/data" --env-file .env bioacoustics-sr python TransformerDetection.py
```

**Note:** Make sure your `.env` file contains `HF_TOKEN=your_token_here` and that any required input files (e.g., audio files in the `data/` directory) exist before running scripts.

## Using EcoVAD for Speech Detection

The project includes `EcoVADDetection.py`, a script that applies segmentation using [ecoVAD](https://github.com/NINAnor/ecoVAD), an end-to-end pipeline for training and using VAD models in soundscape analysis.

### Setup EcoVAD for Docker Container

Since the project directory is mounted as a volume in the Docker container, clone ecoVAD inside the project directory:

**1. Clone ecoVAD repository:**
```bash
# From the project root directory
git clone https://github.com/NINAnor/ecoVAD.git
```

**2. Download model weights:**
1. Visit [OSF](https://osf.io/f4mt5/) and download `assets.zip`
2. Extract to `ecoVAD/assets/` directory inside your project

**3. Install ecoVAD dependencies in the container:**

You can either install dependencies inside the running container, or add them to your project's `requirements.txt` if ecoVAD uses standard pip packages.

**Option A: Install in running container**
```bash
# Enter the running container
docker-compose exec bioacoustics bash

# Navigate to ecoVAD directory
cd /app/ecoVAD

# Install dependencies (if using Poetry - requires poetry to be installed first)
poetry install --no-root

# Or install via pip (if ecoVAD has requirements.txt)
pip install -r requirements.txt
```

**Option B: Rebuild container with ecoVAD dependencies**
If ecoVAD has a `requirements.txt` or `pyproject.toml`, you may need to install its dependencies by adding them to your project's requirements or rebuilding the container.

### Running EcoVADDetection.py

The script works with the same Docker container as your other scripts. It will:
- Automatically detect ecoVAD if cloned in the project directory or parent directory
- Try to use ecoVAD directly if the repository is cloned and model weights are available
- Look for ecoVAD JSON detection files if you've run ecoVAD's `anonymise_data.py` script
- Produce the same output format as `TransformerDetection.py`

**Run the script:**
```bash
docker-compose exec bioacoustics python EcoVADDetection.py
```

**Or run as a one-off command:**
```bash
docker-compose run --rm bioacoustics python EcoVADDetection.py
```

**Note:** The script will fail with clear instructions if ecoVAD is not available. Make sure to set up ecoVAD first (see Setup EcoVAD above).

### Output Files

The script generates:
- `output_with_voice_ecovad.wav` - Audio containing only detected speech segments
- `output_with_silence_ecovad.wav` - Audio with speech segments removed (silence only)

## Known Issues and Solutions

### Interval Detection Beyond Audio Duration

**Issue:** The pyannote.audio segmentation model (`pyannote/segmentation-3.0`) processes audio in fixed 10-second chunks. When processing the final chunk of an audio file, the model may generate detection intervals that extend slightly beyond the actual audio file duration. This occurs because:

1. The model processes chunks of fixed size (10 seconds) and calculates timestamps relative to chunk boundaries
2. The last chunk may be shorter than 10 seconds, but the model treats it as a full chunk
3. Timestamps are calculated from frame positions within chunks, not from the exact audio file duration
4. Small discrepancies can arise from rounding during resampling and format conversion

**Example:** For an audio file with duration 605.863 seconds, the model might detect an interval ending at 605.894 seconds, causing an index out of bounds error when trying to access audio samples beyond the file's actual length.

**Solution:** The detection scripts (`TransformerDetection.py` and `WebRTCVADDetection.py`) now include automatic bounds checking that:

- Clamps all detected intervals to the actual audio file duration before processing
- Validates intervals in both seconds and milliseconds to prevent index errors
- Displays warning messages when intervals are adjusted
- Filters out invalid intervals (where end ≤ start) after clamping

This ensures robust processing even when the VAD model generates timestamps that slightly exceed the audio file boundaries.

### PyTorch 2.6 Compatibility Issue

**Issue:** PyTorch 2.6 changed the default behavior of `torch.load()` to use `weights_only=True` for security. This causes an error when loading pyannote.audio models because the model checkpoints contain `torch.torch_version.TorchVersion` which is not in the default allowlist.

**Error Message:** `_pickle.UnpicklingError: Weights only load failed... Unsupported global: GLOBAL torch.torch_version.TorchVersion`

**Solution:** PyTorch is pinned to version `<2.6.0` in `requirements.txt` to maintain compatibility with pyannote.audio. 

**Note:** `torchaudio` and `pyannote.audio` installation workaround for Python 3.13:
- `torchaudio` requires an exact version match with `torch` (e.g., `torchaudio 2.6.0` requires `torch==2.6.0`)
- `torchaudio 2.5.x` doesn't have Python 3.13 builds
- `torchaudio 2.6.0+` requires `torch 2.6.0+`, which conflicts with our `torch<2.6.0` constraint
- However, `pyannote.audio>=2.1` requires `torchaudio>=2.2.0`
- **Solution:** 
  - `torchaudio` is installed in the Dockerfile with `--no-deps` flag to bypass version checking
  - `torchaudio 2.6.0+` is mostly compatible with `torch 2.5.x` for the operations needed by pyannote.audio
  - Base dependencies are installed first from `requirements-base.txt`, then `pyannote.audio==4.0.3` is installed separately to avoid dependency resolution conflicts

If you encounter this error:

1. Ensure your `requirements.txt` has `torch<2.6.0`
2. The Dockerfile automatically installs `torchaudio` with `--no-deps` to work around the version conflict
3. Rebuild your Docker container: `docker-compose down && docker-compose up --build`
4. If using a local environment, install manually: `pip install --no-deps torchaudio` after installing torch

This issue will be resolved once pyannote.audio updates to support PyTorch 2.6+.

## Notes

- All dependencies (including FFmpeg) are automatically installed in the container
- The project directory is mounted as a volume, so changes to your code are immediately reflected
- Output files (`.wav`, etc.) will be saved in your local project directory
- Make sure to add `.env` to your `.gitignore` file to avoid committing your Hugging Face token
- The Docker setup uses Python 3.13, which is required for audioop-lts and compatible with all project dependencies
- EcoVAD is specifically designed for eco-acoustic data and may perform better on natural soundscapes than general-purpose VAD models
