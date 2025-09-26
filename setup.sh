#!/bin/bash
set -e

echo "🚀 Setting up Voice Assistant..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv (modern Python package manager) is required but not found."
    echo "   Installing uv automatically..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv. Please install manually:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo "✅ uv installed successfully!"
fi

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
    echo "📦 Creating virtual environment with uv..."
    uv venv
fi

# Install requirements with uv
echo "📥 Installing Python packages with uv..."
uv sync

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models

# Download Kokoro TTS models
echo "🎵 Downloading Kokoro TTS models..."
if [ ! -f "models/kokoro-v1.0.onnx" ]; then
    echo "  Downloading kokoro-v1.0.onnx (325MB)..."
    curl -L -o models/kokoro-v1.0.onnx "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
else
    echo "  kokoro-v1.0.onnx already exists, skipping..."
fi

if [ ! -f "models/voices-v1.0.bin" ]; then
    echo "  Downloading voices-v1.0.bin (28MB)..."
    curl -L -o models/voices-v1.0.bin "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
else
    echo "  voices-v1.0.bin already exists, skipping..."
fi

# Preload Whisper ASR model
echo "🎤 Preloading Whisper ASR model..."
echo "  This will download the small.en model (~244MB) to cache for faster startup..."
uv run python -c "
import sys
try:
    from faster_whisper import WhisperModel
    print('  📥 Downloading whisper small.en model...')
    # Download and cache the model - faster_whisper will handle the download
    model = WhisperModel('small.en', device='cpu', compute_type='int8')
    print('  ✅ Whisper small.en model cached successfully!')
except ImportError:
    print('  ⚠️ faster-whisper not installed yet, model will download on first use')
except Exception as e:
    print(f'  ⚠️ Model preload failed: {e}. Model will download on first use.')
" || echo "  ⚠️ Model preload failed, model will download on first use"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Install Ollama for LLM backend:"
echo "   curl -fsSL https://ollama.ai/install.sh | sh"
echo "   ollama serve"
echo "   ollama pull gemma3:270m"
echo ""
echo "2. Run the assistant:"
echo "   uv run python voice_assistant.py"
echo ""
echo "🌐 The web interface will be available at: https://localhost:7860"
echo ""
echo "📁 Downloaded models:"
echo "   - models/kokoro-v1.0.onnx (Kokoro TTS model)"
echo "   - models/voices-v1.0.bin (Voice models including af_heart)"
echo "   - Whisper small.en model (cached for ASR)"
echo ""
echo "🔧 For uv users:"
echo "   uv add package-name     # Add dependencies"
echo "   uv sync                 # Sync dependencies"
echo "   uv run command          # Run in project environment"
echo ""
echo "⚠️  If you encountered build errors:"
echo "   - The assistant will still work with Whisper ASR"
echo "   - SenseVoice ASR is optional for enhanced accuracy"
echo "   - All core TTS functionality is preserved"
