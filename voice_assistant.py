#!/usr/bin/env python3
"""
Real-time Voice Assistant with FastRTC WebRTC Streaming
Enhanced with FastRTC's built-in Silero VAD and Whisper ASR
"""

import gradio as gr
import numpy as np
import threading
import logging
import os
import requests
from typing import Optional, Tuple, Union
import re
from fastrtc import Stream, ReplyOnPause
from fastrtc.pause_detection.silero import SileroVadOptions
import librosa

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperASR:
    """Simple Whisper ASR implementation for CPU-only operation"""

    def __init__(self):
        self.model = None
        self.model_loaded = False

    def _load_model(self):
        """Load Whisper model on CPU"""
        if self.model_loaded:
            return True

        try:
            from faster_whisper import WhisperModel
            logger.info("🎯 ASR: Loading Whisper model (small) on CPU...")
            # Use small model for better accuracy than tiny, but still fast on CPU
            self.model = WhisperModel(
                "small.en", device="cpu", compute_type="int8")
            self.model_loaded = True
            logger.info("🎯 ASR: Whisper model loaded successfully on CPU")
            return True
        except Exception as e:
            logger.error(f"🎯 ASR: Failed to load Whisper model: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio using Whisper"""
        logger.info(
            f"🎯 ASR: Starting transcription of {len(audio)} samples at {sample_rate}Hz")

        if not self.model_loaded:
            logger.info("🎯 ASR: Model not loaded, attempting to load...")
            if not self._load_model():
                logger.error("🎯 ASR: Failed to load model")
                return ""

        try:
            # Ensure audio is float32 and in correct range
            if audio.dtype != np.float32:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                else:
                    audio = audio.astype(np.float32)

            # Normalize audio to [-1, 1] range if needed
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
                logger.info(f"🎯 ASR: Normalized audio from max {max_val:.3f}")

            # Check minimum duration
            duration = len(audio) / sample_rate
            logger.info(
                f"🎯 ASR: Audio duration: {duration:.3f}s, max amplitude: {max_val:.3f}")

            if duration < 0.1:
                logger.warning(
                    f"🎯 ASR: Audio too short ({duration:.3f}s), returning empty")
                return ""

            # Run Whisper transcription
            segments, info = self.model.transcribe(
                audio,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300)
            )

            # Extract text from segments
            transcript = ""
            for segment in segments:
                transcript += segment.text
            transcript = transcript.strip()

            logger.info(
                f"🎯 ASR: Transcription result: '{transcript}' (language confidence: {info.language_probability:.3f})")
            return transcript

        except Exception as e:
            logger.error(f"🎯 ASR: Transcription error: {e}")
            import traceback
            logger.error(f"🎯 ASR: Full traceback: {traceback.format_exc()}")
            return ""


class SimpleVoiceAssistant:
    def __init__(self):
        logger.info("Starting Real-time Voice Assistant with FastRTC")

        self.whisper_model = None
        self.tts_model = None
        self.models_ready = False
        self.conversation_context = []
        self.latest_status = "Auto-initializing models..."
        self.latest_conversation = "# Real-time Voice Assistant with Enhanced VAD/ASR\n\nModels are loading automatically..."

        # Initialize ASR
        self.asr = WhisperASR()

        # Configure VAD options for FastRTC
        self.vad_options = SileroVadOptions(
            threshold=0.5,                    # Speech probability threshold
            min_speech_duration_ms=250,       # Minimum speech duration
            max_speech_duration_s=30.0,       # Maximum speech duration
            # Minimum silence to end speech (reduced from 2000ms)
            min_silence_duration_ms=500,
            # VAD window size (512, 1024, or 1536)
            window_size_samples=1024,
            speech_pad_ms=200                 # Padding around speech
        )

        # Fallback Whisper model
        self.whisper_model = None

        logger.info("Voice Assistant created - will auto-initialize models")
        self._start_auto_initialization()

    def _start_auto_initialization(self):
        def init_thread():
            try:
                logger.info("Auto-initializing models...")
                self.latest_status = "Loading AI models automatically..."

                success, message = self.initialize_models()
                if success:
                    self.latest_status = "Ready! Start speaking..."
                    self.latest_conversation = "# Voice Assistant Ready!\n\nStart speaking now!"
                    logger.info("Auto-initialization complete!")
                else:
                    self.latest_status = "Auto-init failed: " + message
                    self.latest_conversation = "**Error:** " + message
                    logger.error("Auto-initialization failed: " + message)
            except Exception as e:
                logger.error("Auto-initialization error: " + str(e))
                self.latest_status = "Auto-init error: " + str(e)

        thread = threading.Thread(target=init_thread, daemon=True)
        thread.start()

    def initialize_models(self):
        try:
            logger.info("Starting model initialization...")

            logger.info("Loading fallback Whisper model...")
            try:
                from faster_whisper import WhisperModel
                self.whisper_model = WhisperModel("tiny.en", device="cpu")
                logger.info("Whisper fallback model loaded")
            except Exception as e:
                logger.warning(f"Whisper fallback model failed to load: {e}")

            logger.info("Loading Kokoro TTS...")
            try:
                # Compatibility fix for phonemizer/EspeakWrapper
                try:
                    from phonemizer.backend.espeak.wrapper import EspeakWrapper
                    if not hasattr(EspeakWrapper, 'set_data_path'):
                        @classmethod
                        def set_data_path(cls, path):
                            cls.data_path = path
                        EspeakWrapper.set_data_path = set_data_path
                        logger.info(
                            "✅ Fixed EspeakWrapper.set_data_path compatibility issue")

                    # Also fix set_library method if needed
                    if not hasattr(EspeakWrapper, 'set_library'):
                        @classmethod
                        def set_library(cls, path):
                            cls.lib_path = path
                        EspeakWrapper.set_library = set_library
                        logger.info(
                            "✅ Fixed EspeakWrapper.set_library compatibility issue")

                except ImportError:
                    logger.warning(
                        "Could not import EspeakWrapper for compatibility fix")

                import soundfile as sf
                from kokoro_onnx import Kokoro, EspeakConfig

                # Initialize Kokoro TTS with custom EspeakConfig to handle data path issues
                try:
                    # Try with default system espeak
                    espeak_config = EspeakConfig(
                        lib_path="/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
                        data_path="/usr/share/espeak-ng-data"
                    )
                    self.kokoro = Kokoro(
                        "models/kokoro-v1.0.onnx", "models/voices-v1.0.bin", espeak_config=espeak_config)
                except Exception:
                    # Fallback to default configuration
                    logger.info("Trying default Kokoro configuration...")
                    self.kokoro = Kokoro(
                        "models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

                self.tts_backend = "kokoro"
                logger.info("Kokoro TTS loaded successfully")
            except Exception as e:
                logger.error("Kokoro TTS import failed: " + str(e))
                logger.error(f"Kokoro TTS full error: {repr(e)}")
                import traceback
                logger.error(f"Kokoro TTS traceback: {traceback.format_exc()}")
                return False, "Kokoro TTS import failed: " + str(e)

            # Load TTS model
            logger.info("Loading TTS model...")
            try:
                # Kokoro is now loaded, mark as ready
                logger.info("Kokoro TTS ready")
                self.tts_model = "kokoro_ready"
            except Exception as e:
                logger.error("TTS loading failed: " + str(e))
                return False, "TTS loading failed: " + str(e)

            logger.info("Testing Ollama connection...")
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "gemma3:270m",
                          "prompt": "test", "stream": False},
                    timeout=10
                )
                if response.status_code == 200:
                    logger.info("Ollama connection working")
                    self.models_ready = True
                    return True, "All models initialized successfully!"
                else:
                    logger.error("Ollama returned status " +
                                 str(response.status_code))
                    return False, "Ollama not responding"
            except Exception as e:
                logger.error("Ollama connection failed: " + str(e))
                return False, "Ollama connection failed: " + str(e)

        except Exception as e:
            logger.error("Model initialization failed: " + str(e))
            return False, "Initialization error: " + str(e)

    def voice_response(self, audio: tuple[int, np.ndarray]):
        """Process audio input and generate voice response - FastRTC with ReplyOnPause"""
        try:
            sample_rate, audio_data = audio

            logger.info(
                f"🎤 ReplyOnPause: Received {len(audio_data)} samples at {sample_rate}Hz from VAD")

            # ReplyOnPause should have already done VAD processing, so audio_data should be clean speech
            # Ensure the audio is in the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Flatten if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            # Check if we have reasonable speech duration
            duration = len(audio_data) / sample_rate
            logger.info(f"🎤 ReplyOnPause: Speech duration: {duration:.3f}s")

            if duration < 0.1:
                logger.info("🎤 ReplyOnPause: Audio too short, skipping")
                return

            # Resample to 16kHz if needed for ASR
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            # Transcribe with ASR (VAD already done by ReplyOnPause)
            transcript = self.asr.transcribe(audio_data, sample_rate)

            if not transcript or len(transcript.strip()) <= 1:
                logger.info("🎤 ReplyOnPause: No valid transcript generated")
                return

            # Filter out noise patterns
            noise_patterns = [".", "you", "oh",
                              "uh", "um", "a", "i", "the", ""]
            if transcript.lower().strip() in noise_patterns:
                logger.info(
                    f"🎤 ReplyOnPause: Filtered out noise pattern: '{transcript}'")
                return

            logger.info(f"📝 Transcription: '{transcript}'")

            # Generate text response
            response = self.generate_response(transcript)
            logger.info(f"🤖 Response: '{response}'")

            # Update conversation display
            self.latest_conversation += f"\n\n**You:** {transcript}\n\n**Assistant:** {response}"

            # Generate TTS response
            tts_result = self.generate_tts(response)

            if tts_result:
                audio_array, tts_sample_rate = tts_result
                # Yield audio for streaming back
                yield (tts_sample_rate, audio_array.reshape(1, -1))
            else:
                logger.warning("No TTS audio generated")

        except Exception as e:
            logger.error(f"Voice response error: {e}")
            import traceback
            logger.error(f"Voice response traceback: {traceback.format_exc()}")

    def generate_tts(self, text):
        """Generate TTS audio using Kokoro with simple text input (no G2P required)"""
        if not self.models_ready or not self.tts_model:
            logger.warning("TTS not ready")
            return None

        try:
            logger.info("Generating TTS for: '" + text + "'")

            # Generate audio using Kokoro with simple text input and af_heart voice
            # This uses Kokoro's built-in text processing (no manual G2P needed)
            samples, sample_rate = self.kokoro.create(
                text, voice="af_heart", speed=1.0, lang="en-us"
            )

            # Ensure audio is float32 and in the right range
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            # Normalize if needed (Kokoro usually outputs in [-1, 1] range)
            if np.max(np.abs(samples)) > 1.0:
                samples = samples / np.max(np.abs(samples))

            # Resample to 16kHz if needed (for WebRTC compatibility)
            if sample_rate != 16000:
                import librosa
                samples = librosa.resample(
                    samples, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            logger.info("TTS generated: " +
                        str(len(samples)) + " samples at " + str(sample_rate) + " Hz")

            return samples, sample_rate

        except Exception as e:
            logger.error("TTS generation error: " + str(e))
            import traceback
            logger.error(f"TTS generation traceback: {traceback.format_exc()}")
            return None

    def get_status(self):
        """Get current status"""
        if self.models_ready:
            status_msg = "🟢 Ready for real-time conversation!"
        else:
            status_msg = self.latest_status

        return status_msg, self.latest_conversation

    def create_interface(self):
        with gr.Blocks(title="Real-time Voice Assistant") as interface:
            gr.Markdown("# 🎙️ Real-time WebRTC Voice Assistant")
            gr.Markdown(
                "**Ultra-low latency voice conversation with professional audio processing**\n\n"
                "• **Silero VAD**: Advanced voice activity detection\n"
                "• **Whisper ASR**: High-accuracy speech recognition\n"
                "• **Kokoro TTS**: High-quality neural text-to-speech with af_heart voice\n"
                "• **FastRTC WebRTC**: Ultra-low latency streaming"
            )

            # Main WebRTC Voice Assistant Interface

            status_display = gr.Textbox(
                label="Status",
                value=self.latest_status,
                interactive=False
            )

            conversation_display = gr.Markdown(
                value=self.latest_conversation)

            # Create WebRTC stream with ReplyOnPause and VAD options
            stream = Stream(
                modality="audio",
                mode="send-receive",
                handler=ReplyOnPause(
                    self.voice_response,
                    model_options=self.vad_options
                )
            )

            # Status refresh timer
            status_refresh = gr.Timer(value=1.0)
            status_refresh.tick(
                fn=self.get_status,
                inputs=[],
                outputs=[status_display, conversation_display]
            )

        return interface

    def generate_response(self, text):
        """
        Generate text response
        Your task is to implement the openai gpt-4.1-mini model
        """

        if not text.strip():
            return "I didn't catch that. Could you repeat?"

        try:
            self.conversation_context.append("User: " + text)

            if len(self.conversation_context) > 20:
                self.conversation_context = self.conversation_context[-20:]

            context = "\n".join(self.conversation_context)

            prompt = "You are a helpful voice assistant. Be concise and natural in your responses.\n\nConversation:\n" + \
                context + "\n\nResponse as assistant:"

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:270m",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 150
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get('response', '').strip()

                self.conversation_context.append(
                    "Assistant: " + assistant_response)

                logger.info("LLM Response: '" + assistant_response + "'")
                return assistant_response
            else:
                logger.error("Ollama request failed: " +
                             str(response.status_code))
                return "Sorry, I'm having trouble thinking right now."

        except Exception as e:
            logger.error("Response generation error: " + str(e))
            return "Sorry, I encountered an error processing your request."


if __name__ == "__main__":
    logger.info("Starting Simple Voice Assistant")

    assistant = SimpleVoiceAssistant()
    interface = assistant.create_interface()

    logger.info("Launching interface with HTTPS for WebRTC support...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssl_keyfile="ssl_certs/key.pem",
        ssl_certfile="ssl_certs/cert.pem",
        ssl_verify=False  # Allow self-signed certificates
    )
