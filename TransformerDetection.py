
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
import os
from huggingface_hub import login
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline
from dotenv import load_dotenv

load_dotenv()

# Read your Hugging Face token from environment. Do NOT hardcode secrets.
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in your environment."
    )

# Optionally ensure the token is active for the session (no-op if already logged in)
try:
    login(HF_TOKEN, add_to_git_credential=False)
except Exception:
    pass

# Model identifier for VAD (pyannote 3.1-compatible)
# Use the segmentation model directly instead of the pipeline
from pyannote.audio import Model

model_name = "pyannote/segmentation-3.0"
model = Model.from_pretrained(model_name, token=HF_TOKEN)

# Create a simple VAD pipeline using the segmentation model
from pyannote.audio.pipelines import VoiceActivityDetection

pipeline = VoiceActivityDetection(segmentation=model)

# Instantiate the pipeline with default parameters
HYPER_PARAMETERS = {
    "min_duration_on": 0.1,
    "min_duration_off": 0.1,
}
pipeline.instantiate(HYPER_PARAMETERS)

# Function to load and preprocess audio
def load_and_preprocess_audio(audio_path):
    """Loads and preprocesses audio for VAD.  Resamples to 16kHz mono, converts to NumPy array."""
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None, None, None

    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Resample to 16kHz
    samples = np.array(audio.get_array_of_samples())
    samples = samples.astype(np.float32) / np.iinfo(np.int16).max  # Normalize
    return samples, audio, audio.frame_rate

# Function to perform VAD
def detect_speech_intervals(samples, sample_rate, pipeline):
    """Performs voice activity detection on the audio."""

    try:
        # The pipeline expects a dictionary with 'waveform' and 'sample_rate'
        input_data = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": sample_rate}
        vad_result = pipeline(input_data)
    except Exception as e:
        print(f"Error during VAD: {e}")
        return [], 0.0

    intervals = []
    for segment in vad_result.get_timeline():
        intervals.append((segment.start, segment.end))
    return intervals, 0.0

# Function to save audio with intervals
def save_audio_with_intervals(audio, intervals, output_with_voice, output_with_silence, duration):
    silent_audio = AudioSegment.silent(duration=len(audio))

    audio_with_voice = silent_audio
    audio_with_silence = audio

    for start, end in intervals:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        audio_with_voice = audio_with_voice.overlay(audio[start_ms:end_ms], position=start_ms)
        audio_with_silence = audio_with_silence.overlay(silent_audio[start_ms:end_ms], position=start_ms)

    audio_with_voice.export(output_with_voice, format="wav")
    audio_with_silence.export(output_with_silence, format="wav")

# Paths
# Update this to a valid local path on your machine
mp3_path = os.path.join("data","XC240120 - Soundscape.mp3") 

# Validate input path early
if not os.path.isfile(mp3_path):
    raise FileNotFoundError(
        f"Audio file not found: {mp3_path}. Update 'mp3_path' to a valid local file."
    )
output_with_voice = "output_with_voice_huggingface.wav"
output_with_silence = "output_with_silence_huggingface.wav"

# Load and preprocess audio
samples, audio_segment, sample_rate = load_and_preprocess_audio(mp3_path)

# Detect speech intervals
speech_intervals, duration = detect_speech_intervals(samples, sample_rate, pipeline)
duration = len(audio_segment) / 1000.0

print(speech_intervals)

# Save new audio files with intervals
save_audio_with_intervals(audio_segment, speech_intervals, output_with_voice, output_with_silence, duration)
