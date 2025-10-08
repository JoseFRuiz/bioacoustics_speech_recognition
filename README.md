# bioacoustics_speech_recognition

## Environment Setup (Conda + Pip)

Follow these steps to create an isolated Conda environment and install project dependencies from `requirements.txt`.

1) Create a new Conda environment (choose a name you like, e.g., `bioacoustics-sr`). You can also pin a Python version if needed (example uses 3.10):

```bash
conda create -n bioacoustics-sr python=3.10 -y
```

2) Activate the environment:

```bash
conda activate bioacoustics-sr
```

3) Upgrade `pip` (recommended) and install dependencies from `requirements.txt` located at the repo root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4) Install

```bash   
conda install -c conda-forge ffmpeg
```

Notes:
- Ensure `requirements.txt` exists in the repository (usually at the root).
- If you prefer a different Python version, adjust the `python=` spec accordingly or omit it to use Conda’s default.
- On first use of Conda in a new shell, you may need to run `conda init` and restart the terminal.
