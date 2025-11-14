import webrtcvad
import collections
import contextlib
import wave
import os
import subprocess
from pydub import AudioSegment

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000)  # Resample to 16000 Hz
    audio = audio.set_channels(1)  # Ensure audio is mono
    audio.export(wav_path, format="wav")

# Frame class to hold audio data
class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    voiced_frames = []
    intervals = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                intervals.append((ring_buffer[0][0].timestamp, None))
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                intervals[-1] = (intervals[-1][0], frame.timestamp + frame.duration)
                triggered = False
                ring_buffer.clear()

    if triggered:
        intervals[-1] = (intervals[-1][0], frame.timestamp + frame.duration)

    return intervals

def save_audio_with_intervals(wav_path, intervals, output_with_voice, output_with_silence):
    audio = AudioSegment.from_wav(wav_path)
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
mp3_path = os.path.join("data","XC240120 - Soundscape.mp3")  # Upload your MP3 file to Colab
wav_path = "temp_audio.wav"
output_with_voice = "output_with_voice_WebRTC.wav"
output_with_silence = "output_with_silence_WebRTC.wav"

# Convert MP3 to WAV
convert_mp3_to_wav(mp3_path, wav_path)

# Read WAV file
audio, sample_rate = read_wave(wav_path)

# Initialize VAD
vad = webrtcvad.Vad(3)

# Generate frames
frames = frame_generator(30, audio, sample_rate)
frames = list(frames)

# Collect voiced segments
intervals = vad_collector(sample_rate, 30, 300, vad, frames)

# Print intervals
for start, end in intervals:
    print(f"Start: {start:.2f}s, End: {end:.2f}s")

# Save new audio files with intervals
save_audio_with_intervals(wav_path, intervals, output_with_voice, output_with_silence)

# Cleanup temporary WAV file
os.remove(wav_path)
