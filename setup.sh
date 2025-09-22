#!/bin/bash
set -e

echo "🚀 Setting up Voice Assistant..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Install system dependencies for compilation
echo "🔧 Installing system dependencies..."
if command -v dnf &> /dev/null; then
    # Fedora/RHEL
    echo "  Installing build tools for Fedora/RHEL..."
    sudo dnf install -y gcc gcc-c++ python3-devel espeak espeak-devel
elif command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    echo "  Installing build tools for Ubuntu/Debian..."
    sudo apt-get update
    sudo apt-get install -y build-essential python3-dev espeak espeak-data libespeak-dev
elif command -v yum &> /dev/null; then
    # CentOS/older RHEL
    echo "  Installing build tools for CentOS..."
    sudo yum install -y gcc gcc-c++ python3-devel espeak espeak-devel
else
    echo "⚠️  Could not detect package manager. You may need to install build tools manually:"
    echo "   - gcc/g++ compiler"
    echo "   - python3-dev/python3-devel"
    echo "   - espeak and espeak-devel"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip, setuptools, and wheel
echo "⬆️ Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install requirements with better error handling
echo "📥 Installing Python packages..."
if ! pip install -r requirements.txt; then
    echo "❌ Some packages failed to install. Trying with alternative approaches..."
    
    # Try installing problematic packages individually
    echo "🔧 Installing core packages first..."
    pip install numpy
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Try installing misaki with no-build-isolation to avoid blis issues
    echo "🔧 Installing misaki with build workaround..."
    pip install --no-build-isolation misaki[en] || pip install misaki
    
    # Install remaining packages
    echo "🔧 Installing remaining packages..."
    pip install gradio fastapi uvicorn fastrtc librosa soundfile onnxruntime faster-whisper funasr kokoro-onnx phonemizer requests ollama asyncio websockets pydub loguru
fi

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models

# Download Kokoro TTS models
echo "🎵 Downloading Kokoro TTS models..."
if [ ! -f "models/kokoro-v1.0.onnx" ]; then
    echo "  Downloading kokoro-v1.0.onnx (325MB)..."
    curl -L -o models/kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/v1.0/kokoro-v1.0.onnx"
else
    echo "  kokoro-v1.0.onnx already exists, skipping..."
fi

if [ ! -f "models/voices-v1.0.bin" ]; then
    echo "  Downloading voices-v1.0.bin (28MB)..."
    curl -L -o models/voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/v1.0/voices-v1.0.bin"
else
    echo "  voices-v1.0.bin already exists, skipping..."
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Make sure Ollama is installed and running:"
echo "   curl -fsSL https://ollama.ai/install.sh | sh"
echo "   ollama serve"
echo "   ollama pull gemma3:270m"
echo ""
echo "2. Activate the environment and run the assistant:"
echo "   source .venv/bin/activate"
echo "   python3 voice_assistant_ascii.py"
echo ""
echo "🌐 The web interface will be available at: http://localhost:7860"
echo ""
echo "📁 Downloaded models:"
echo "   - models/kokoro-v1.0.onnx (Kokoro TTS model)"
echo "   - models/voices-v1.0.bin (Voice models including af_heart)"
echo ""
echo "⚠️  If you encountered build errors:"
echo "   - The assistant will still work with Whisper ASR"
echo "   - SenseVoice ASR is optional for enhanced accuracy"
echo "   - All core TTS functionality is preserved"
