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

**Rebuild after dependency changes:**
```bash
docker-compose up --build
```

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

## Notes

- All dependencies (including FFmpeg) are automatically installed in the container
- The project directory is mounted as a volume, so changes to your code are immediately reflected
- Output files (`.wav`, etc.) will be saved in your local project directory
- Make sure to add `.env` to your `.gitignore` file to avoid committing your Hugging Face token
- The Docker setup uses Python 3.13, which is required for audioop-lts and compatible with all project dependencies
