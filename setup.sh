#!/bin/bash
set -e

echo "🚀 Setting up Bob Voice Assistant..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install basic requirements
echo "📥 Installing basic Python packages..."
pip install gradio fastapi uvicorn[standard] numpy torch torchaudio loguru

# Install AI/ML packages
echo "🧠 Installing AI/ML packages..."
pip install ollama faster-whisper

# Try to install Kokoro TTS
echo "🗣️ Installing TTS packages..."
if pip install kokoro; then
    echo "✅ Kokoro TTS installed successfully"
else
    echo "⚠️ Kokoro TTS installation failed, trying alternative..."
    pip install pyttsx3 gtts  # Fallback TTS options
fi

# Install optional packages
echo "🔧 Installing optional packages..."
pip install scipy websockets pydub

# Try to install gradio-webrtc if available
echo "🌐 Attempting to install WebRTC support..."
if pip install gradio-webrtc; then
    echo "✅ WebRTC support installed"
else
    echo "⚠️ WebRTC support not available, using standard Gradio audio"
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
echo "   python3 simple_voice_assistant.py"
echo ""
echo "3. Or run the advanced version with VAD:"
echo "   python3 advanced_voice_assistant.py"
echo ""
echo "🌐 The web interface will be available at: http://localhost:7860"
