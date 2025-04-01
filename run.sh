#!/usr/bin/env bash
set -e

# --- Create/use persistent virtual environment (venv) ---
if [ ! -d "venv" ]; then
    echo "Creating persistent virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "Using existing virtual environment 'venv'."
fi
source venv/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# --- Check for required external dependencies ---
for cmd in aplay ollama; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: $cmd is not installed or not in PATH."
    echo "Please install $cmd before running this script."
    exit 1
  fi
done

# --- Create/use persistent packages directory ---
if [ ! -d "packages" ]; then
    echo "Creating persistent packages directory 'packages'..."
    mkdir packages
else
    echo "Using existing packages directory 'packages'."
fi

# --- Create/use persistent temporary build directory for pip ---
if [ ! -d "tmp" ]; then
    echo "Creating persistent temporary directory 'tmp' for builds..."
    mkdir tmp
else
    echo "Using existing temporary directory 'tmp'."
fi
export TMPDIR="$(pwd)/tmp"
echo "TMPDIR set to: $(pwd)/tmp"

# --- Create/use persistent IO directory ---
if [ ! -d "io" ]; then
    echo "Creating persistent IO directory 'io'..."
    mkdir io
else
    echo "Using existing IO directory 'io'."
fi

# --- Install required Python packages into 'packages' ---
echo "Installing required Python packages into 'packages' directory..."
pip install git+https://github.com/openai/whisper.git --target=packages
pip install webrtcvad sounddevice numpy==2.1.3 scipy --target=packages

# Add the 'packages' directory to PYTHONPATH so Python finds the dependencies.
export PYTHONPATH="$(pwd)/packages:$PYTHONPATH"
echo "PYTHONPATH set to: $(pwd)/packages"

# --- Install Local Piper, Voice, and Config ---
# URLs provided by you:
PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"
VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx?download=true"
CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json?download=true.json"

# Define installation directories relative to the current folder
PIPER_DIR="./piper"
VOICES_DIR="$PIPER_DIR/voices"
# After extraction, the Piper executable is at ./piper/piper/piper
PIPER_BIN="$PIPER_DIR/piper/piper"
VOICE_FILE="$VOICES_DIR/en_US-ryan-high.onnx"
CONFIG_FILE="$VOICES_DIR/en_US-ryan-high.onnx.json"

if [ ! -f "$PIPER_BIN" ]; then
    echo "Downloading and installing Piper locally..."
    mkdir -p "$PIPER_DIR"
    wget -O piper.tar.gz "$PIPER_URL"
    tar -xzvf piper.tar.gz -C "$PIPER_DIR"
    rm piper.tar.gz
    chmod +x "$PIPER_BIN"
    echo "Piper installed locally in $PIPER_DIR."
else
    echo "Local Piper installation found in $PIPER_DIR."
    chmod +x "$PIPER_BIN"
fi

if [ ! -f "$VOICE_FILE" ]; then
    echo "Downloading high-quality voice file (en_US-ryan-high)..."
    mkdir -p "$VOICES_DIR"
    wget -O "$VOICE_FILE" "$VOICE_URL"
    echo "Voice file downloaded to $VOICE_FILE."
else
    echo "Voice file en_US-ryan-high already exists in $VOICES_DIR."
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Downloading voice configuration file (en_US-ryan-high)..."
    mkdir -p "$VOICES_DIR"
    wget -O "$CONFIG_FILE" "$CONFIG_URL"
    echo "Configuration file downloaded to $CONFIG_FILE."
else
    echo "Configuration file already exists in $VOICES_DIR."
fi

# Add the local Piper directory (the parent of the extracted 'piper' folder) to PATH so that assistant.py finds it.
export PATH="$PWD/piper:$PATH"
echo "Local Piper directory added to PATH: $PWD/piper"

# --- Start the Assistant ---
echo "Starting JAKE Voice Assistant with Ryan high-quality voice..."
if [ -f "./assistant.py" ]; then
    python ./assistant.py
else
    echo "Error: assistant.py not found in the current directory."
fi

echo "Persistent virtual environment, packages, tmp, io, and Piper directories are available."
