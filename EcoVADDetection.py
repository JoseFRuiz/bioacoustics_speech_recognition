"""
EcoVAD-based Voice Activity Detection Script

This script uses ecoVAD to detect speech segments in audio files,
similar to the TransformerDetection.py script but using ecoVAD instead.

Requirements:
- ecoVAD repository should be cloned locally or available in the path
- Model weights from OSF: https://osf.io/f4mt5/ (download assets folder)
- See README for ecoVAD setup instructions

Setup Instructions:
1. Clone ecoVAD repository:
   git clone https://github.com/NINAnor/ecoVAD.git
   
2. Download model weights:
   - Visit https://osf.io/f4mt5/
   - Download assets.zip
   - Extract to ecoVAD/assets/ directory

3. Install ecoVAD dependencies (if using Poetry):
   cd ecoVAD
   poetry install --no-root
"""

from pydub import AudioSegment
import numpy as np
import os
import sys
import json
import csv
import subprocess
from pathlib import Path
import torch

# Try to import ecoVAD modules
ECOVAD_AVAILABLE = False
ecovad_predictor = None

try:
    # Option 1: If ecoVAD repo is cloned locally, add it to path
    ecovad_path = Path(__file__).parent.parent / "ecoVAD"
    if not ecovad_path.exists():
        # Try current directory
        ecovad_path = Path(__file__).parent / "ecoVAD"
    
    if ecovad_path.exists():
        sys.path.insert(0, str(ecovad_path))
        print(f"Found ecoVAD at: {ecovad_path}")
        
        # Try to import ecoVAD modules
        try:
            # Import based on ecoVAD's actual structure
            from VAD_algorithms.ecovad import ecoVADpredict
            ECOVAD_AVAILABLE = True
            print("Successfully imported ecoVAD modules")
        except ImportError as e:
            print(f"Could not import ecoVAD modules: {e}")
            print("Will try to use ecoVAD via JSON detection files")
    else:
        print("ecoVAD repository not found in expected locations")
        print("Please clone it: git clone https://github.com/NINAnor/ecoVAD.git")
        
except Exception as e:
    print(f"Error setting up ecoVAD: {e}")

# Function to load and preprocess audio
def load_and_preprocess_audio(audio_path):
    """Loads and preprocesses audio for VAD. Resamples to 16kHz mono, converts to NumPy array."""
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


def detect_speech_intervals_ecovad(samples, sample_rate, audio_path, model_path=None, threshold=0.7, use_gpu=False):
    """
    Performs voice activity detection using ecoVAD.
    
    Args:
        samples: Audio samples as numpy array (not used directly, audio_path is used instead)
        sample_rate: Sample rate of the audio (not used directly)
        audio_path: Path to the audio file
        model_path: Path to ecoVAD model weights (optional, will try to find default)
        threshold: Confidence threshold for detection (default: 0.7)
        use_gpu: Whether to use GPU (default: False)
    
    Returns:
        intervals: List of (start, end) tuples in seconds
    
    Raises:
        RuntimeError: If ecoVAD is not available or model weights are not found
    """
    intervals = []
    
    # Try to use ecoVAD directly if available
    if ECOVAD_AVAILABLE:
        try:
            # Check for model weights in common locations
            if model_path is None:
                ecovad_base = Path(__file__).parent.parent / "ecoVAD"
                if not ecovad_base.exists():
                    ecovad_base = Path(__file__).parent / "ecoVAD"
                
                possible_paths = [
                    ecovad_base / "assets" / "model_weights" / "ecoVAD_ckpt.pt",
                    ecovad_base / "assets" / "model_weights" / "ecoVAD_model_weight.pt",
                    ecovad_base / "assets" / "model_weights" / "ecoVAD_weights_demo.pt",
                    ecovad_base / "assets" / "ecovad_model.pt",
                    ecovad_base / "assets" / "models" / "ecovad_model.pt",
                    Path("assets/model_weights/ecoVAD_ckpt.pt"),
                    Path("assets/ecovad_model.pt"),
                ]
                for path in possible_paths:
                    if path.exists():
                        model_path = str(path)
                        break
            
            if model_path and os.path.exists(model_path):
                print(f"Using ecoVAD model from: {model_path}")
                
                # Create a temporary output path for JSON file
                import tempfile
                temp_dir = tempfile.mkdtemp()
                output_json_path = os.path.join(temp_dir, "ecovad_detections")
                
                # Initialize ecoVADpredict and run detection
                from VAD_algorithms.ecovad import ecoVADpredict
                predictor = ecoVADpredict(
                    input=str(audio_path),
                    output=output_json_path,
                    model_path=model_path,
                    threshold=threshold,
                    use_gpu=use_gpu
                )
                
                # Patch initDataset to use num_workers=0 to avoid shared memory issues in Docker
                def patched_initDataset():
                    from torch.utils.data import DataLoader
                    
                    list_segments_tensor = [np.array(segment.get_array_of_samples(), dtype=float) for segment in
                                            predictor.soundscapeSegments]
                    list_segments_mel = [predictor.to_mel_spectrogram(segment) for segment in list_segments_tensor]
                    list_segment_mel_norm = [predictor.normalize_row_matrix(segment) for segment in list_segments_mel]
                    # Convert to numpy array first to avoid the warning
                    list_segment_mel_norm = np.array(list_segment_mel_norm)
                    list_segment_mel_norm = torch.tensor(list_segment_mel_norm)
                    list_segment_mel_norm = list_segment_mel_norm.unsqueeze(1)
                    
                    # Use num_workers=0 to avoid shared memory issues
                    predLoader = DataLoader(list_segment_mel_norm,
                                            batch_size=predictor.batch_size,
                                            shuffle=False,
                                            num_workers=0,  # Disable multiprocessing to avoid shm issues
                                            pin_memory=predictor.use_gpu
                                            )
                    return predLoader
                
                predictor.initDataset = patched_initDataset
                predictor.main()
                
                # Parse the generated JSON file
                json_file = output_json_path + ".json"
                if os.path.exists(json_file):
                    intervals = parse_ecovad_json(json_file)
                    # Clean up temporary file
                    try:
                        os.remove(json_file)
                        os.rmdir(temp_dir)
                    except:
                        pass
                else:
                    raise FileNotFoundError(f"ecoVAD did not generate output JSON file: {json_file}")
                    
            else:
                raise FileNotFoundError(
                    "ecoVAD model weights not found.\n"
                    "Please download from: https://osf.io/f4mt5/\n"
                    "Extract assets folder and place model weights in assets/model_weights/ directory."
                )
        
        except Exception as e:
            raise RuntimeError(f"Error using ecoVAD: {e}")
    else:
        raise RuntimeError(
            "ecoVAD is not available.\n"
            "Please set up ecoVAD:\n"
            "1. Clone repository: git clone https://github.com/NINAnor/ecoVAD.git\n"
            "2. Download model weights from: https://osf.io/f4mt5/\n"
            "3. Extract assets folder to ecoVAD/assets/\n"
            "Alternatively, run ecoVAD's anonymise_data.py to generate JSON detection files."
        )
    
    return intervals


def parse_ecovad_json(json_path):
    """
    Parse ecoVAD JSON detection file.
    
    Args:
        json_path: Path to JSON file containing ecoVAD detections
    
    Returns:
        intervals: List of (start, end) tuples in seconds
    """
    intervals = []
    try:
        with open(json_path, 'r') as f:
            detections = json.load(f)
        
        # Parse ecoVAD JSON format: {'ecoVAD': "Timeline", 'content': [{'start': int, 'end': int}, ...]}
        # The start/end values are in seconds (indices of 1-second segments)
        if isinstance(detections, dict):
            if 'content' in detections:
                # Standard ecoVAD format
                for segment in detections['content']:
                    if isinstance(segment, dict):
                        start = float(segment.get('start', segment.get('start_time', 0)))
                        end = float(segment.get('end', segment.get('end_time', 0)))
                        intervals.append((start, end))
            elif 'segments' in detections:
                for segment in detections['segments']:
                    if isinstance(segment, dict):
                        intervals.append((segment.get('start', segment.get('start_time', 0)),
                                        segment.get('end', segment.get('end_time', 0))))
                    elif isinstance(segment, list) and len(segment) >= 2:
                        intervals.append((float(segment[0]), float(segment[1])))
            elif 'detections' in detections:
                for detection in detections['detections']:
                    intervals.append((detection.get('start', 0), detection.get('end', 0)))
            elif 'intervals' in detections:
                intervals = [(iv['start'], iv['end']) for iv in detections['intervals']]
        elif isinstance(detections, list):
            # List of segments
            for segment in detections:
                if isinstance(segment, dict):
                    intervals.append((segment.get('start', segment.get('start_time', 0)),
                                    segment.get('end', segment.get('end_time', 0))))
                elif isinstance(segment, list) and len(segment) >= 2:
                    intervals.append((float(segment[0]), float(segment[1])))
        
        return intervals
    except Exception as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        return []


def detect_speech_intervals_from_file(audio_path, model_path=None):
    """
    Convenience function to detect speech intervals directly from an audio file.
    Uses ecoVAD's anonymise_data.py approach if direct API is not available.
    """
    # First, try to find existing ecoVAD JSON detection files
    audio_name = Path(audio_path).stem
    possible_json_locations = [
        Path(f"assets/demo_data_/detections/json/ecoVAD/{audio_name}.json"),
        Path(f"assets/detections/json/ecoVAD/{audio_name}.json"),
        Path(f"detections/json/ecoVAD/{audio_name}.json"),
    ]
    
    # Check for ecoVAD repository paths
    ecovad_base = Path(__file__).parent.parent / "ecoVAD"
    if not ecovad_base.exists():
        ecovad_base = Path(__file__).parent / "ecoVAD"
    if ecovad_base.exists():
        possible_json_locations.extend([
            ecovad_base / "assets" / "demo_data_" / "detections" / "json" / "ecoVAD" / f"{audio_name}.json",
            ecovad_base / "assets" / "detections" / "json" / "ecoVAD" / f"{audio_name}.json",
        ])
    
    for json_path in possible_json_locations:
        if json_path.exists():
            print(f"Found existing ecoVAD detection file: {json_path}")
            intervals = parse_ecovad_json(json_path)
            if intervals:
                audio_segment = AudioSegment.from_file(audio_path)
                return intervals, audio_segment
    
    # Try direct API
    try:
        samples, audio_segment, sample_rate = load_and_preprocess_audio(audio_path)
        intervals = detect_speech_intervals_ecovad(samples, sample_rate, audio_path, model_path)
        if intervals:
            return intervals, audio_segment
    except Exception as e:
        print(f"Direct API failed: {e}")
        raise
    
    # If all else fails, provide instructions
    raise RuntimeError(
        "Could not use ecoVAD. Please set up ecoVAD:\n"
        "1. Clone ecoVAD repository: git clone https://github.com/NINAnor/ecoVAD.git\n"
        "2. Download model weights from: https://osf.io/f4mt5/\n"
        "3. Extract assets folder to ecoVAD/assets/\n"
        "4. Or run ecoVAD's anonymise_data.py to generate detection JSON files"
    )


def save_audio_with_intervals(audio, intervals, output_with_voice, output_with_silence):
    """Saves audio files with detected intervals."""
    silent_audio = AudioSegment.silent(duration=len(audio))

    audio_with_voice = silent_audio
    audio_with_silence = audio

    for start, end in intervals:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        # Ensure we don't exceed audio bounds
        start_ms = max(0, min(start_ms, len(audio)))
        end_ms = max(0, min(end_ms, len(audio)))
        
        if end_ms > start_ms:
            audio_with_voice = audio_with_voice.overlay(audio[start_ms:end_ms], position=start_ms)
            audio_with_silence = audio_with_silence.overlay(silent_audio[start_ms:end_ms], position=start_ms)

    audio_with_voice.export(output_with_voice, format="wav")
    audio_with_silence.export(output_with_silence, format="wav")
    print(f"Saved audio with voice segments to: {output_with_voice}")
    print(f"Saved audio with silence segments to: {output_with_silence}")


def main():
    """Main function to run ecoVAD detection."""
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths
    audio_code = "XC240120"
    mp3_path = os.path.join("data", f"{audio_code} - Soundscape.mp3")
    
    # Validate input path
    if not os.path.isfile(mp3_path):
        raise FileNotFoundError(
            f"Audio file not found: {mp3_path}. Update 'mp3_path' to a valid local file."
        )
    
    output_with_voice = os.path.join(output_dir, f"output_with_voice_ecovad_{audio_code}.wav")
    output_with_silence = os.path.join(output_dir, f"output_with_silence_ecovad_{audio_code}.wav")
    csv_output_path = os.path.join(output_dir, f"voice_activity_intervals_ecovad_{audio_code}.csv")
    
    print("=" * 60)
    print("EcoVAD Voice Activity Detection")
    print("=" * 60)
    print(f"Input audio: {mp3_path}")
    print()
    
    # Detect speech intervals using ecoVAD
    try:
        speech_intervals, audio_segment = detect_speech_intervals_from_file(mp3_path)
        print("Using ecoVAD for detection")
        
        if speech_intervals is None or len(speech_intervals) == 0:
            print("Warning: No speech intervals detected.")
            print("This could mean:")
            print("  - No speech was found in the audio")
            print("  - The detection model needs adjustment")
            print("  - There was an error in detection")
        else:
            print(f"\nDetected {len(speech_intervals)} speech segments:")
            for i, (start, end) in enumerate(speech_intervals, 1):
                print(f"  Segment {i}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
            
            # Save intervals to CSV file
            with open(csv_output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Start Time (s)', 'End Time (s)', 'Duration (s)'])
                for start, end in speech_intervals:
                    duration_interval = end - start
                    writer.writerow([f'{start:.3f}', f'{end:.3f}', f'{duration_interval:.3f}'])
            
            print(f"\nCSV file saved to: {csv_output_path}")
            
            # Save new audio files with intervals
            save_audio_with_intervals(audio_segment, speech_intervals, output_with_voice, output_with_silence)
            
            print(f"Audio files saved to: {output_dir}")
            print("\n" + "=" * 60)
            print("Detection completed successfully!")
            print("=" * 60)
    except Exception as e:
        print(f"\nEcoVAD detection failed: {e}")
        raise


if __name__ == "__main__":
    main()

