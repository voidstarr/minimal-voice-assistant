#!/usr/bin/env python3
"""
Real-time Voice Assistant with FastRTC WebRTC Streaming
Enhanced with Silero VAD and SenseVoice ASR
"""

import gradio as gr
import numpy as np
import threading
import logging
import os
import requests
from typing import Optional, Tuple, Union
import re
import math
import enum
from fastrtc import Stream, ReplyOnPause
import librosa

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpeakingStatus(enum.Enum):
    PRE_START = enum.auto()
    START = enum.auto()
    END = enum.auto()


class SileroVAD:
    """Silero VAD implementation based on OpenAvatarChat"""
    def __init__(self):
        self.model = None
        self.speaking_threshold = 0.5
        self.start_delay = 2048  # samples
        self.end_delay = 5000    # samples
        self.buffer_look_back = 1024
        self.speech_padding = 512
        self.clip_size = 512
        
        # State tracking
        self.speaking_status = SpeakingStatus.END
        self.audio_history = []
        self.speech_length = 0
        self.silence_length = 0
        self.model_state = None
        self.speech_id = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load Silero VAD ONNX model"""
        try:
            import onnxruntime
            model_path = "/home/foss/code/mini-ai-demo/models/silero_vad.onnx"
            
            if not os.path.exists(model_path):
                logger.error(f"VAD model not found at {model_path}")
                return
                
            options = onnxruntime.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            options.log_severity_level = 4
            self.model = onnxruntime.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
                sess_options=options
            )
            self.model_state = np.zeros((2, 1, 128), dtype=np.float32)
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
    
    def _inference(self, clip: np.ndarray, sr: int = 16000):
        """Run VAD inference on audio clip"""
        if self.model is None:
            return 0.0
            
        clip = clip.squeeze()
        if clip.ndim != 1:
            logger.warning("Input audio should be 1-dim array")
            return 0.0
            
        clip = np.expand_dims(clip, axis=0)
        inputs = {
            "input": clip,
            "sr": np.array([sr], dtype=np.int64),
            "state": self.model_state
        }
        
        try:
            prob, state = self.model.run(None, inputs)
            self.model_state = state
            return prob[0][0]
        except Exception as e:
            logger.error(f"VAD inference error: {e}")
            return 0.0
    
    def _append_to_history(self, clip: np.ndarray):
        """Add audio clip to history buffer"""
        self.audio_history.append(clip)
        history_limit = math.ceil((self.start_delay + self.buffer_look_back) / self.clip_size)
        while len(self.audio_history) > history_limit:
            self.audio_history.pop(0)
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Process audio chunk and return speech audio when detected"""
        if len(audio_chunk) < self.clip_size:
            # Pad short chunks
            padded = np.zeros(self.clip_size, dtype=audio_chunk.dtype)
            padded[:len(audio_chunk)] = audio_chunk
            audio_chunk = padded
        
        # Split into clips of the right size
        clips = []
        for i in range(0, len(audio_chunk), self.clip_size):
            clip = audio_chunk[i:i+self.clip_size]
            if len(clip) < self.clip_size:
                # Pad the last clip
                padded = np.zeros(self.clip_size, dtype=clip.dtype)
                padded[:len(clip)] = clip
                clip = padded
            clips.append(clip)
        
        speech_audio = None
        
        for clip in clips:
            self._append_to_history(clip)
            
            # Run VAD inference
            speech_prob = self._inference(clip)
            
            # Update speech/silence counters
            if speech_prob > self.speaking_threshold:
                self.speech_length += self.clip_size
                self.silence_length = 0
            else:
                self.silence_length += self.clip_size
                self.speech_length = 0
            
            # State machine for speech detection
            if self.speaking_status == SpeakingStatus.END:
                if self.speech_length > 0:
                    logger.debug("Pre-start of new human speech")
                    self.speaking_status = SpeakingStatus.PRE_START
                    
            elif self.speaking_status == SpeakingStatus.PRE_START:
                if self.speech_length >= self.start_delay:
                    self.speaking_status = SpeakingStatus.START
                    self.speech_id += 1
                    
                    # Collect buffered audio for speech start
                    sample_num_to_fetch = self.buffer_look_back + self.start_delay
                    clip_num_to_fetch = math.ceil(sample_num_to_fetch / self.clip_size)
                    
                    speech_clips = self.audio_history[-clip_num_to_fetch:]
                    if speech_clips:
                        speech_audio = np.concatenate(speech_clips, axis=0)
                        # Add padding
                        padding = np.zeros(self.speech_padding, dtype=speech_audio.dtype)
                        speech_audio = np.concatenate([padding, speech_audio], axis=0)
                        logger.info("🎤 Speech started")
                elif self.silence_length > 0:
                    logger.debug("Back to not started status")
                    self.speaking_status = SpeakingStatus.END
                    
            elif self.speaking_status == SpeakingStatus.START:
                if self.silence_length >= self.end_delay:
                    self.speaking_status = SpeakingStatus.END
                    # Add final clip with padding
                    padding = np.zeros(self.speech_padding, dtype=clip.dtype)
                    final_audio = np.concatenate([clip, padding], axis=0)
                    logger.info("🔇 Speech ended")
                    return final_audio
                else:
                    # Continue collecting speech audio
                    speech_audio = clip
        
        return speech_audio


class SenseVoiceASR:
    """SenseVoice ASR implementation based on OpenAvatarChat"""
    def __init__(self):
        self.model = None
        self.model_name = 'iic/SenseVoiceSmall'
        self.model_loaded = False
    
    def _load_model(self):
        """Lazy load SenseVoice ASR model"""
        if self.model_loaded:
            return True
            
        try:
            from funasr import AutoModel
            logger.info("Loading SenseVoice ASR model (this may take a while)...")
            self.model = AutoModel(model=self.model_name, disable_update=True)
            self.model_loaded = True
            logger.info("SenseVoice ASR model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load SenseVoice ASR: {e}")
            return False
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio using SenseVoice"""
        if not self.model_loaded:
            if not self._load_model():
                return ""
        
        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                else:
                    audio = audio.astype(np.float32)
            
            # SenseVoice expects 16kHz audio
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Run transcription
            result = self.model.generate(input=audio.squeeze(), batch_size_s=10)
            
            if result and len(result) > 0:
                # Remove special tokens
                text = re.sub(r"<\|.*?\|>", "", result[0]['text'])
                return text.strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
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

        # Initialize VAD and ASR
        self.vad = SileroVAD()
        self.asr = SenseVoiceASR()
        
        # Fallback Whisper model
        self.whisper_model = None
        
        # Audio accumulation buffer for complete utterances
        self.speech_buffer = []
        self.is_collecting_speech = False

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
                # Compatibility fix for phonemizer
                try:
                    from phonemizer.backend.espeak.wrapper import EspeakWrapper
                    if not hasattr(EspeakWrapper, 'set_data_path'):
                        @classmethod
                        def set_data_path(cls, path):
                            cls.espeak_library = path
                        EspeakWrapper.set_data_path = set_data_path
                        print("✅ Fixed EspeakWrapper.set_data_path compatibility issue")
                except ImportError:
                    # No wrapper available, that's fine
                    pass
                
                import soundfile as sf
                from misaki import en, espeak
                from kokoro_onnx import Kokoro
                
                # Initialize G2P (Grapheme-to-Phoneme)
                fallback = espeak.EspeakFallback(british=False)
                self.g2p = en.G2P(trf=False, british=False, fallback=fallback)
                
                # Initialize Kokoro TTS
                self.kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")
                self.tts_backend = "kokoro"
                logger.info("Kokoro TTS loaded successfully")
            except Exception as e:
                logger.error("Kokoro TTS import failed: " + str(e))
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

    def process_audio_with_vad_asr(self, audio_data: np.ndarray, sample_rate: int) -> Optional[str]:
        """Process audio with VAD and ASR"""
        try:
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_data = audio_data.astype(np.float32)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=16000
                )
            
            # Process through VAD
            speech_audio = self.vad.process_audio_chunk(audio_data)
            
            if speech_audio is not None:
                if self.vad.speaking_status == SpeakingStatus.START:
                    # Start collecting speech
                    if not self.is_collecting_speech:
                        self.is_collecting_speech = True
                        self.speech_buffer = [speech_audio]
                        logger.debug("Started collecting speech")
                    else:
                        self.speech_buffer.append(speech_audio)
                        
                elif self.vad.speaking_status == SpeakingStatus.END and self.is_collecting_speech:
                    # End of speech - process complete utterance
                    self.speech_buffer.append(speech_audio)
                    complete_speech = np.concatenate(self.speech_buffer, axis=0)
                    
                    logger.info(f"Processing complete speech: {len(complete_speech)} samples")
                    
                    # Transcribe with SenseVoice
                    transcript = self.asr.transcribe(complete_speech, 16000)
                    
                    # Fallback to Whisper if SenseVoice fails
                    if not transcript and self.whisper_model:
                        logger.info("Falling back to Whisper ASR")
                        try:
                            segments, info = self.whisper_model.transcribe(
                                complete_speech,
                                beam_size=5,
                                language="en",
                                condition_on_previous_text=False
                            )
                            transcript = ""
                            for segment in segments:
                                transcript += segment.text
                            transcript = transcript.strip()
                        except Exception as e:
                            logger.error(f"Whisper fallback error: {e}")
                    
                    # Reset collection state
                    self.is_collecting_speech = False
                    self.speech_buffer = []
                    
                    if transcript and len(transcript.strip()) > 1:
                        # Filter out noise patterns
                        noise_patterns = [".", "you", "oh", "uh", "um", "a", "i", "the", ""]
                        if transcript.lower().strip() not in noise_patterns:
                            return transcript.strip()
                    
                elif self.is_collecting_speech:
                    # Continue collecting
                    self.speech_buffer.append(speech_audio)
            
            return None
            
        except Exception as e:
            logger.error(f"VAD+ASR processing error: {e}")
            return None

    def voice_response(self, audio: tuple[int, np.ndarray]):
        """Process audio input and generate voice response - FastRTC compatible"""
        try:
            sample_rate, audio_data = audio

            # Ensure the audio is in the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Flatten if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            logger.debug(f"🎤 Processing {len(audio_data)} audio samples at {sample_rate}Hz")

            # Process with enhanced VAD+ASR
            transcript = self.process_audio_with_vad_asr(audio_data, sample_rate)

            if not transcript:
                logger.debug("No transcript generated")
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
                audio_array, sample_rate = tts_result
                # Yield audio for streaming back
                yield (sample_rate, audio_array.reshape(1, -1))
            else:
                logger.warning("No TTS audio generated")

        except Exception as e:
            logger.error(f"Voice response error: {e}")

    def generate_response(self, text):
        if not text.strip():
            return "I didn't catch that. Could you repeat?"

        try:
            self.conversation_context.append("User: " + text)

            if len(self.conversation_context) > 6:
                self.conversation_context = self.conversation_context[-6:]

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

    def generate_tts(self, text):
        """Generate TTS audio using Kokoro with af_heart voice"""
        if not self.models_ready or not self.tts_model:
            logger.warning("TTS not ready")
            return None

        try:
            logger.info("Generating TTS for: '" + text + "'")

            # Convert text to phonemes using G2P
            phonemes, _ = self.g2p(text)
            
            # Generate audio using Kokoro with af_heart voice
            samples, sample_rate = self.kokoro.create(phonemes, "af_heart", is_phonemes=True)
            
            # Ensure audio is float32 and in the right range
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            
            # Normalize if needed (Kokoro usually outputs in [-1, 1] range)
            if np.max(np.abs(samples)) > 1.0:
                samples = samples / np.max(np.abs(samples))
            
            # Resample to 16kHz if needed (for WebRTC compatibility)
            if sample_rate != 16000:
                import librosa
                samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000

            logger.info("TTS generated: " +
                            str(len(samples)) + " samples at " + str(sample_rate) + " Hz")

            return samples, sample_rate

        except Exception as e:
            logger.error("TTS generation error: " + str(e))
            return None

    def get_status(self):
        """Get current status"""
        if self.models_ready:
            status_msg = "🟢 Ready for real-time conversation!"
        else:
            status_msg = self.latest_status

        return status_msg, self.latest_conversation

    def create_interface(self):
        with gr.Blocks(title="Enhanced Voice Assistant - VAD+ASR") as interface:
            gr.Markdown("# 🎙️ Enhanced Voice Assistant with Silero VAD + SenseVoice ASR")
            gr.Markdown(
                "**Ultra-reliable voice processing with professional VAD and ASR!**\n\n"
                "• **Silero VAD**: Advanced voice activity detection\n"
                "• **SenseVoice ASR**: High-accuracy speech recognition\n"
                "• **Kokoro TTS**: High-quality neural text-to-speech with af_heart voice\n"
                "• **FastRTC WebRTC**: Ultra-low latency streaming"
            )

            status_display = gr.Textbox(
                label="Status",
                value=self.latest_status,
                interactive=False
            )

            conversation_display = gr.Markdown(value=self.latest_conversation)

            # Create WebRTC stream with ReplyOnPause
            stream = Stream(
                modality="audio",
                mode="send-receive",
                handler=ReplyOnPause(self.voice_response)
            )

            # Status refresh timer
            status_refresh = gr.Timer(value=1.0)
            status_refresh.tick(
                fn=self.get_status,
                inputs=[],
                outputs=[status_display, conversation_display]
            )

        return interface


if __name__ == "__main__":
    logger.info("Starting Simple Voice Assistant")

    assistant = SimpleVoiceAssistant()
    interface = assistant.create_interface()

    logger.info("Launching interface with HTTPS for WebRTC support...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssl_keyfile="/home/foss/code/mini-ai-demo/ssl_certs/key.pem",
        ssl_certfile="/home/foss/code/mini-ai-demo/ssl_certs/cert.pem",
        ssl_verify=False  # Allow self-signed certificates
    )
