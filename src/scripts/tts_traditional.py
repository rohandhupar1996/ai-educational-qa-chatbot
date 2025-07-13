"""
Traditional TTS Models - gTTS and basic TTS implementations
"""
import os
import logging
from typing import Optional
from gtts import gTTS
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    slow: bool = False
    output_filename: Optional[str] = None

class TTSResponse(BaseModel):
    success: bool
    output_file: Optional[str] = None
    error: Optional[str] = None

class GoogleTTS:
    """
    Google Text-to-Speech wrapper using gTTS library.
    """
    
    def __init__(self, default_lang: str = "en", default_slow: bool = False):
        """
        Initialize Google TTS.
        
        Args:
            default_lang: Default language code
            default_slow: Whether to use slow speech by default
        """
        self.default_lang = default_lang
        self.default_slow = default_slow
        logger.info("Initialized Google TTS")
    
    def synthesize(self, text: str, lang: str = None, slow: bool = None, output_file: str = None) -> TTSResponse:
        """
        Convert text to speech using Google TTS.
        
        Args:
            text: Text to convert to speech
            lang: Language code (e.g., 'en', 'es', 'fr')
            slow: Whether to use slow speech
            output_file: Output file path
            
        Returns:
            TTSResponse with success status and output file path
        """
        try:
            if not text or not text.strip():
                return TTSResponse(
                    success=False,
                    error="Text cannot be empty"
                )
            
            # Use defaults if not provided
            lang = lang or self.default_lang
            slow = slow if slow is not None else self.default_slow
            output_file = output_file or f"gtts_output_{hash(text) % 10000}.mp3"
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
            
            # Create and save TTS
            tts = gTTS(text=text, lang=lang, slow=slow)
            tts.save(output_file)
            
            logger.info(f"Audio saved as {output_file}")
            
            return TTSResponse(
                success=True,
                output_file=output_file
            )
            
        except Exception as e:
            logger.error(f"Error in Google TTS: {e}")
            return TTSResponse(
                success=False,
                error=f"Google TTS error: {str(e)}"
            )

def gtts_tts(text: str, lang: str = 'en', output_file: str = "gTTS_output.mp3") -> str:
    """
    Convert text to speech using gTTS and save the output as an audio file.
    
    Args:
        text: The text to be converted to speech
        lang: The language of the text. Default is 'en' (English)
        output_file: The name of the output audio file
        
    Returns:
        Path to the generated audio file
        
    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text for gTTS cannot be empty.")
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_file)
        print(f"Audio saved as {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error in gtts_tts: {e}")
        raise

# Available languages for gTTS
GTTS_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'hi': 'Hindi',
    'ar': 'Arabic'
}

def get_supported_languages() -> dict:
    """Get dictionary of supported language codes and names."""
    return GTTS_LANGUAGES.copy()

def validate_language(lang_code: str) -> bool:
    """Check if language code is supported."""
    return lang_code in GTTS_LANGUAGES