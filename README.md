# bioacoustics_speech_recognition

## Environment Setup (Conda + Pip)

Install Microsoft C++ Build Tools (Recommended)
The most straightforward solution is to install the Microsoft C++ Build Tools:
Download and install Microsoft C++ Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
During installation, make sure to select "C++ build tools" workload
After installation, restart your terminal and try installing again:


Follow these steps to create an isolated Conda environment and install project dependencies from `requirements.txt`.

1) Create a new Conda environment (choose a name you like, e.g., `bioacoustics-sr`). You can also pin a Python version if needed (example uses 3.13):

```bash
conda create -n bioacoustics-sr python=3.13 -y
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
