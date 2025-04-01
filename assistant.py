import sys
import os
# Insert the 'packages' directory (in the same folder as this script) into sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages"))

import sounddevice as sd
import numpy as np
import queue
import collections
import time
import subprocess
import whisper
from scipy.io.wavfile import write
import webrtcvad
import argparse
import logging
import shutil
import stat
import sys

# Use the IO directory (created by the bash script) for sound files.
IO_DIR = os.path.join(os.getcwd(), "io")
INPUT_WAV = os.path.join(IO_DIR, "input.wav")
OUTPUT_WAV = os.path.join(IO_DIR, "output.wav")

# ---------------- Argument Parsing ----------------
def parse_args():
    parser = argparse.ArgumentParser(description="JAKE Voice Assistant")
    parser.add_argument("--vad-mode", type=int, default=1, choices=[0, 1, 2, 3],
                        help="VAD aggressiveness (0-3); use a higher number (e.g., 3) to ignore softer sounds")
    # Increased default silence duration to 1000 ms.
    parser.add_argument("--silence-duration", type=int, default=1000,
                        help="Silence duration in ms to consider utterance ended")
    parser.add_argument("--piper-voice", type=str, default="en_US-ryan-high",
                        help="Piper voice to use")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate in Hz")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size (tiny, base, small, etc.)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    return parser.parse_args()

# ---------------- Configuration ----------------
args = parse_args()

# Set up logging
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("JAKE")

SAMPLE_RATE = args.sample_rate
CHANNELS = 1               # mono audio
FRAME_DURATION = 30        # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # samples per frame

# TTS voice for Piper; we now use the full path if available.
PIPER_VOICE = args.piper_voice

# VAD configuration
VAD_MODE = args.vad_mode
SILENCE_DURATION_MS = args.silence_duration
SILENCE_FRAMES = int(SILENCE_DURATION_MS / FRAME_DURATION)

# Ollama model
OLLAMA_MODEL = "phi4-mini"

# ---------------- Global Objects ----------------
audio_q = queue.Queue()

# ---------------- Piper Path Management ----------------
def get_piper_path():
    """
    Force use of the local Piper executable.
    Our extraction creates a structure: ./piper/piper/piper
    """
    local_piper = os.path.join(os.getcwd(), "piper", "piper", "piper")
    if os.path.exists(local_piper) and os.access(local_piper, os.X_OK):
        return local_piper
    piper_in_path = shutil.which("piper")
    if piper_in_path:
        return piper_in_path
    return "piper"

# ---------------- Piper Voice Path Management ----------------
def get_voice_path(voice_name):
    """
    Return the full path to the voice model file.
    """
    local_voice = os.path.join(os.getcwd(), "piper", "voices", voice_name + ".onnx")
    if os.path.exists(local_voice):
        return local_voice
    return voice_name

# ---------------- Audio Callback ----------------
def audio_callback(indata, frames, time_info, status):
    if status:
        logger.warning(f"Audio callback status: {status}")
    data = np.frombuffer(indata, dtype=np.float32).reshape(-1, CHANNELS)
    audio_q.put(data.copy())

# ---------------- Utterance Recording ----------------
def record_utterance():
    vad = webrtcvad.Vad(VAD_MODE)
    voiced_frames = []
    triggered = False
    silence_counter = 0
    ring_buffer = collections.deque(maxlen=SILENCE_FRAMES)
    logger.info("Waiting for speech...")
    while True:
        try:
            frame = audio_q.get(timeout=1)
        except queue.Empty:
            continue
        frame = frame.flatten()
        frame_int16 = (frame * 32767).astype(np.int16)
        frame_bytes = frame_int16.tobytes()
        try:
            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            continue
        if not triggered:
            ring_buffer.append(frame)
            if is_speech:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
                logger.info("Speech detected, recording...")
        else:
            voiced_frames.append(frame)
            silence_counter = silence_counter + 1 if not is_speech else 0
            if silence_counter > SILENCE_FRAMES:
                logger.info("Silence detected, processing utterance...")
                break
            if len(voiced_frames) > 60 * (SAMPLE_RATE // FRAME_SIZE):
                logger.warning("Maximum recording length reached, processing utterance...")
                break
    if voiced_frames:
        return np.concatenate(voiced_frames)
    else:
        return None

# ---------------- Whisper Transcription ----------------
def transcribe_audio(filename, model):
    logger.info("Transcribing audio...")
    try:
        result = model.transcribe(filename)
        transcription = result["text"].strip()
        logger.info(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

# ---------------- Query Ollama ----------------
def query_ollama(prompt_text):
    logger.info(f"Querying Ollama ({OLLAMA_MODEL})...")
    system_prompt = ("You are JAKE, a helpful voice assistant powered by phi4-mini. "
                     "Keep responses concise, natural, and conversational. Your name is JAKE.")
    full_prompt = f"system: {system_prompt}\nuser: {prompt_text}"
    command = ["ollama", "run", OLLAMA_MODEL, full_prompt]
    try:
        output = subprocess.check_output(command, text=True)
        response = output.strip()
        logger.info(f"JAKE Response: {response}")
        return response
    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling Ollama: {e}")
        return "I'm having trouble processing that request. Please try again."

# ---------------- Piper TTS Synthesis ----------------
def synthesize_tts(text, output_file):
    logger.info("Synthesizing speech with Piper...")
    piper_path = get_piper_path()
    voice_path = get_voice_path(PIPER_VOICE)
    logger.debug(f"Using Piper at: {piper_path}")
    logger.debug(f"Using voice model: {voice_path}")
    if not os.access(piper_path, os.X_OK):
        logger.error(f"Permission denied: {piper_path} is not executable.")
        return
    command = [piper_path, "--voice", voice_path, "--text", text, "--output", output_file]
    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        logger.info(f"Speech synthesized and saved to {output_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in Piper TTS: {e}")
        logger.error(f"Error output: {e.stderr.decode()}")
        try:
            subprocess.run(["espeak", "-w", output_file, text], check=True)
            logger.info("Used espeak as fallback TTS")
        except Exception as fallback_e:
            logger.error("Failed to use fallback TTS")

# ---------------- Play Audio ----------------
def play_audio(filename):
    logger.info("Playing audio...")
    try:
        subprocess.run(["aplay", filename], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error playing audio: {e}")

# ---------------- Main Loop ----------------
def main():
    logger.info("Starting JAKE Voice Assistant. Press Ctrl+C to exit.")
    for cmd in ["aplay", "ollama"]:
        try:
            subprocess.run(["which", cmd], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            logger.error(f"Error: {cmd} is not installed or not in PATH")
            return
    piper_path = get_piper_path()
    if piper_path != os.path.join(os.getcwd(), "piper", "piper", "piper"):
        logger.warning(f"Using non-local Piper at {piper_path}. To force local usage, remove legacy installations.")
    try:
        logger.info(f"Loading Whisper model ({args.whisper_model})...")
        whisper_model = whisper.load_model(args.whisper_model)
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.error("Exiting due to model initialization failure")
        return
    logger.info("Warming up Ollama model...")
    startup_response = query_ollama("initialize")
    logger.info(f"Ollama warm-up response: {startup_response}")
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE, dtype='float32',
                                 channels=CHANNELS, callback=audio_callback):
            logger.info("Audio stream started. JAKE is listening...")
            while True:
                try:
                    utterance = record_utterance()
                    if utterance is None:
                        logger.warning("No utterance detected, listening again...")
                        continue
                    write(INPUT_WAV, SAMPLE_RATE, utterance)
                    logger.debug(f"Utterance recorded to {INPUT_WAV}")
                    transcription = transcribe_audio(INPUT_WAV, whisper_model)
                    if not transcription:
                        logger.warning("No transcription available, skipping...")
                        continue
                    response_text = query_ollama(transcription)
                    if not response_text:
                        logger.warning("No response from Ollama, skipping...")
                        continue
                    synthesize_tts(response_text, OUTPUT_WAV)
                    play_audio(OUTPUT_WAV)
                    # Pause a bit and flush the audio queue to reduce self-triggering
                    time.sleep(1.5)
                    while not audio_q.empty():
                        try:
                            audio_q.get_nowait()
                        except queue.Empty:
                            break
                    if os.path.exists(INPUT_WAV):
                        os.remove(INPUT_WAV)
                    if os.path.exists(OUTPUT_WAV):
                        os.remove(OUTPUT_WAV)
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    logger.info("Exiting JAKE assistant gracefully.")
                    sys.exit(0)
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    logger.info("Continuing to listen...")
    except KeyboardInterrupt:
        logger.info("Exiting JAKE assistant gracefully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting audio stream: {e}")

if __name__ == "__main__":
    main()
