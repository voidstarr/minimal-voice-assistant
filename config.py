"""
Configuration settings for Voice Assistant - Final Edition
"""

# Model Configuration
WHISPER_MODEL = "tiny.en"  # Options: tiny, base, small, medium, large
LLM_MODEL = "gemma3:270m"  # Ollama model name
# EdgeTTS voice (en-US-AriaNeural, en-US-JennyNeural, etc.)
TTS_VOICE = "en-US-AriaNeural"
TTS_SPEED = 1.2            # TTS playback speed

# Audio Configuration
SAMPLE_RATE = 16000        # Sample rate for audio processing
CHUNK_SIZE = 1024          # Audio chunk size for processing
CHANNELS = 1               # Number of audio channels (mono)

# Voice Activity Detection
VAD_THRESHOLD = 0.01       # RMS threshold for voice detection
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration (seconds)
MIN_SILENCE_DURATION = 1.0  # Minimum silence before processing (seconds)
MAX_RECORDING_DURATION = 15.0  # Maximum recording duration (seconds)

# Server Configuration
HOST = "0.0.0.0"
PORT = 7860
SHARE = False              # Set to True to create public shareable link

# Processing Configuration
MAX_CONVERSATION_HISTORY = 6  # Number of previous exchanges to remember
RESPONSE_TIMEOUT = 30      # Timeout for AI response generation (seconds)
AUDIO_BUFFER_DURATION = 10.0  # Maximum audio buffer duration (seconds)

# UI Configuration
THEME = "soft"             # Gradio theme: default, soft, monochrome, etc.
TITLE = "🤖 Voice Assistant"
DESCRIPTION = """
A voice assistant using local AI models for speech recognition, 
natural language processing, and text-to-speech synthesis.
"""

# Model Initialization Messages
SYSTEM_PROMPT = """
You are John, a helpful and friendly voice assistant. 
Keep your responses concise and conversational, typically 1-2 sentences.
Be helpful, engaging, and maintain a warm personality.
You can help with questions, have conversations, and assist with various tasks.
"""

# Error Messages
ERROR_MESSAGES = {
    "transcription_failed": "Sorry, I couldn't understand what you said. Please try speaking more clearly.",
    "llm_failed": "I'm having trouble generating a response right now. Please try again.",
    "tts_failed": "I can't speak right now, but I can still help you with text responses.",
    "audio_processing_failed": "There was an issue processing your audio. Please try again.",
    "model_loading_failed": "Failed to load AI models. Please check your configuration.",
    "no_audio": "No audio detected. Please make sure your microphone is working.",
    "timeout": "Response timeout. Please try with a shorter message."
}

# Success Messages
SUCCESS_MESSAGES = {
    "models_loaded": "✅ All AI models loaded successfully!",
    "transcription_complete": "🎧 Audio transcribed successfully",
    "response_generated": "🤖 Response generated successfully",
    "speech_synthesized": "🗣️ Speech synthesis completed",
    "ready": "🚀 John Voice Assistant is ready!"
}

# Feature Flags
FEATURES = {
    "conversation_memory": True,    # Remember conversation context
    "voice_activity_detection": True,  # Automatic speech detection
    "audio_normalization": True,   # Normalize audio levels
    "error_recovery": True,        # Attempt to recover from errors
    "debug_logging": False,        # Enable debug logging
    "save_audio_files": False,     # Save audio files for debugging
}
