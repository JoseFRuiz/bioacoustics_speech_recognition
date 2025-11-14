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

Using Docker directly:

```bash
# Build the Docker image
docker build -t bioacoustics-sr .

# Run the container
docker run -p 8888:8888 \
  -v "${PWD}:/app" \
  -v "${PWD}/data:/app/data" \
  --env-file .env \
  bioacoustics-sr
```

**3) Access Jupyter Notebook**

Once the container is running, open your browser and navigate to:
- `http://localhost:8888`

The Jupyter notebook interface will be available. Open `bioacoustics_speech_recognition.ipynb` to start working.

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

**Using Docker directly:**
```bash
# Rebuild the image
docker build -t bioacoustics-sr .

# Or rebuild without cache
docker build --no-cache -t bioacoustics-sr .

# Then run the container
docker run -p 8888:8888 \
  -v "${PWD}:/app" \
  -v "${PWD}/data:/app/data" \
  --env-file .env \
  bioacoustics-sr
```

**Note:** After rebuilding, you may need to restart your container if it was already running. Use `docker-compose down` followed by `docker-compose up` to ensure you're using the newly built image.

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

## Notes

- All dependencies (including FFmpeg) are automatically installed in the container
- The project directory is mounted as a volume, so changes to your code are immediately reflected
- Output files (`.wav`, etc.) will be saved in your local project directory
- Make sure to add `.env` to your `.gitignore` file to avoid committing your Hugging Face token
- The Docker setup uses Python 3.13, which is required for audioop-lts and compatible with all project dependencies
- EcoVAD is specifically designed for eco-acoustic data and may perform better on natural soundscapes than general-purpose VAD models
