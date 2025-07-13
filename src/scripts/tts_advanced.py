"""
Advanced TTS Models - Tacotron, OuteTTS, Glow-TTS, YourTTS
"""
import os
import logging
from typing import Optional, List
from pydantic import BaseModel
import IPython.display as ipd
from IPython.display import Audio, display

logger = logging.getLogger(__name__)

class AdvancedTTSRequest(BaseModel):
    text: str
    model_type: str
    output_filename: Optional[str] = None
    # Model-specific parameters
    speed: float = 1.2
    temperature: float = 0.1
    repetition_penalty: float = 1.1
    speaker_name: str = "male_2"
    speaker_index: int = 0
    language_index: int = 0
    sample_rate: int = 22050

class AdvancedTTSResponse(BaseModel):
    success: bool
    output_file: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None

class TacotronTTS:
    """
    Tacotron2 TTS model wrapper.
    """
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", use_gpu: bool = False):
        self.use_gpu = use_gpu  # Set to False for macOS CPU
        """
        Initialize Tacotron TTS.
        
        Args:
            model_name: Pre-trained model name
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        logger.info(f"Initialized Tacotron TTS with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the TTS model."""
        if self.model is None:
            try:
                from TTS.api import TTS
                self.model = TTS(model_name=self.model_name, gpu=self.use_gpu)
                logger.info("Tacotron model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Tacotron model: {e}")
                raise
    
    def synthesize(self, text: str, output_file: str = None, speed: float = 1.2) -> AdvancedTTSResponse:
        """
        Convert text to speech using Tacotron.
        
        Args:
            text: Text to convert
            output_file: Output file path
            speed: Speech speed
            
        Returns:
            AdvancedTTSResponse
        """
        try:
            if not text or not text.strip():
                return AdvancedTTSResponse(
                    success=False,
                    error="Text cannot be empty"
                )
            
            self._load_model()
            
            output_file = output_file or f"tacotron_output_{hash(text) % 10000}.wav"
            
            # Generate audio
            self.model.tts_to_file(
                text=text,
                file_path=output_file,
                speed=speed
            )
            
            logger.info(f"Tacotron audio saved as {output_file}")
            
            return AdvancedTTSResponse(
                success=True,
                output_file=output_file,
                model_used="Tacotron2"
            )
            
        except Exception as e:
            logger.error(f"Error in Tacotron TTS: {e}")
            return AdvancedTTSResponse(
                success=False,
                error=f"Tacotron TTS error: {str(e)}",
                model_used="Tacotron2"
            )

class OuteTTS:
    """
    OuteTTS model wrapper.
    """
    
    def __init__(self):
        """Initialize OuteTTS."""
        self.interface = None
        logger.info("Initialized OuteTTS")
    
    def _load_model(self):
        if self.interface is None:
            try:
                from outetts.v0_1.interface import InterfaceGGUF
                self.interface = InterfaceGGUF("OuteTTS-0.1-350M")
            except Exception as e:
                logger.error(f"Failed to load OuteTTS interface: {e}")
                raise
        
    def synthesize(self, text: str, output_file: str = None, temperature: float = 0.1, 
                   repetition_penalty: float = 1.1, speaker_name: str = "male_2") -> AdvancedTTSResponse:
        """
        Convert text to speech using OuteTTS.
        
        Args:
            text: Text to convert
            output_file: Output file path
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty
            speaker_name: Speaker name
            
        Returns:
            AdvancedTTSResponse
        """
        try:
            if not text or not text.strip():
                return AdvancedTTSResponse(
                    success=False,
                    error="Text cannot be empty"
                )
            
            self._load_model()
            
            output_file = output_file or f"outetts_output_{hash(text) % 10000}.wav"
            
            # Load speaker
            speaker = self.interface.load_default_speaker(name=speaker_name)
            
            # Generate speech
            output = self.interface.generate(
                text=text,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                speaker=speaker,
            )
            
            # Save audio
            output.save(output_file)
            
            logger.info(f"OuteTTS audio saved as {output_file}")
            
            return AdvancedTTSResponse(
                success=True,
                output_file=output_file,
                model_used="OuteTTS"
            )
            
        except Exception as e:
            logger.error(f"Error in OuteTTS: {e}")
            return AdvancedTTSResponse(
                success=False,
                error=f"OuteTTS error: {str(e)}",
                model_used="OuteTTS"
            )

class GlowTTS:
    """
    Glow-TTS model wrapper.
    """
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/glow-tts"):
        """
        Initialize Glow-TTS.
        
        Args:
            model_name: Pre-trained model name
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initialized Glow-TTS with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the TTS model."""
        if self.model is None:
            try:
                from TTS.api import TTS
                self.model = TTS(model_name=self.model_name)
                logger.info("Glow-TTS model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Glow-TTS model: {e}")
                raise
    
    def synthesize(self, text: str, output_file: str = None, sample_rate: int = 22050) -> AdvancedTTSResponse:
        """
        Convert text to speech using Glow-TTS.
        
        Args:
            text: Text to convert
            output_file: Output file path
            sample_rate: Audio sample rate
            
        Returns:
            AdvancedTTSResponse
        """
        try:
            if not text or not text.strip():
                return AdvancedTTSResponse(
                    success=False,
                    error="Text cannot be empty"
                )
            
            self._load_model()
            
            output_file = output_file or f"glow_tts_output_{hash(text) % 10000}.wav"
            
            # Generate audio
            audio_output = self.model.tts(text)
            
            # Save audio
            self.model.save_wav(audio_output, output_file)
            
            logger.info(f"Glow-TTS audio saved as {output_file}")
            
            return AdvancedTTSResponse(
                success=True,
                output_file=output_file,
                model_used="Glow-TTS"
            )
            
        except Exception as e:
            logger.error(f"Error in Glow-TTS: {e}")
            return AdvancedTTSResponse(
                success=False,
                error=f"Glow-TTS error: {str(e)}",
                model_used="Glow-TTS"
            )

class YourTTS:
    """
    YourTTS model wrapper.
    """
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/your_tts"):
        """
        Initialize YourTTS.
        
        Args:
            model_name: Pre-trained model name
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initialized YourTTS with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the TTS model."""
        if self.model is None:
            try:
                from TTS.api import TTS
                self.model = TTS(model_name=self.model_name)
                logger.info("YourTTS model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YourTTS model: {e}")
                raise
    
    def synthesize(self, text: str, output_file: str = None, speaker_index: int = 0, 
                   language_index: int = 0) -> AdvancedTTSResponse:
        """
        Convert text to speech using YourTTS.
        
        Args:
            text: Text to convert
            output_file: Output file path
            speaker_index: Speaker index
            language_index: Language index
            
        Returns:
            AdvancedTTSResponse
        """
        try:
            if not text or not text.strip():
                return AdvancedTTSResponse(
                    success=False,
                    error="Text cannot be empty"
                )
            
            self._load_model()
            
            output_file = output_file or f"your_tts_output_{hash(text) % 10000}.wav"
            
            # Get available speakers and languages
            available_speakers = self.model.speakers if hasattr(self.model, 'speakers') else []
            available_languages = self.model.languages if hasattr(self.model, 'languages') else []
            
            # Select speaker and language
            selected_speaker = available_speakers[speaker_index] if available_speakers else None
            selected_language = available_languages[language_index] if available_languages else None
            
            # Generate audio
            self.model.tts_to_file(
                text=text, 
                speaker=selected_speaker, 
                language=selected_language, 
                file_path=output_file
            )
            
            logger.info(f"YourTTS audio saved as {output_file}")
            
            return AdvancedTTSResponse(
                success=True,
                output_file=output_file,
                model_used="YourTTS"
            )
            
        except Exception as e:
            logger.error(f"Error in YourTTS: {e}")
            return AdvancedTTSResponse(
                success=False,
                error=f"YourTTS error: {str(e)}",
                model_used="YourTTS"
            )

# Convenience functions matching the original paste.txt format
def tacotron_tts(text: str, output_file: str = "tacotron_audio.wav", speed: float = 1.2) -> str:
    """Convert text to speech using Tacotron."""
    if not text:
        raise ValueError("Text for Tacotron cannot be empty.")
    
    tacotron = TacotronTTS()
    response = tacotron.synthesize(text, output_file, speed)
    
    if response.success:
        print(f"Audio generation completed. File saved as '{output_file}'.")
        return output_file
    else:
        raise Exception(response.error)

def outetts_tts(text: str, output_file: str = "OuteTTS_audio.wav", temperature: float = 0.1, 
                repetition_penalty: float = 1.1, speaker_name: str = "male_2") -> str:
    """Convert text to speech using OuteTTS."""
    if not text:
        raise ValueError("Text for OuteTTS cannot be empty.")
    
    outetts = OuteTTS()
    response = outetts.synthesize(text, output_file, temperature, repetition_penalty, speaker_name)
    
    if response.success:
        print(f"Audio generation completed. File saved as '{output_file}'.")
        return output_file
    else:
        raise Exception(response.error)

def glow_tts(text: str, output_file: str = "glow_tts_output.wav", sample_rate: int = 22050) -> str:
    """Convert text to speech using Glow-TTS."""
    if not text:
        raise ValueError("Text for Glow-TTS cannot be empty.")
    
    glow = GlowTTS()
    response = glow.synthesize(text, output_file, sample_rate)
    
    if response.success:
        print(f"Audio generation completed. File saved as '{output_file}'.")
        return output_file
    else:
        raise Exception(response.error)

def your_tts(text: str, speaker_index: int = 0, language_index: int = 0, 
             output_file: str = "YourTTS_output.wav") -> str:
    """Convert text to speech using YourTTS."""
    if not text:
        raise ValueError("Text for YourTTS cannot be empty.")
    
    yourtts = YourTTS()
    response = yourtts.synthesize(text, output_file, speaker_index, language_index)
    
    if response.success:
        print(f"Audio generation completed. File saved as '{output_file}'.")
        return output_file
    else:
        raise Exception(response.error)