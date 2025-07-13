"""
Audio processing utilities and helper functions.
"""
import os
import logging
from typing import Optional, Tuple
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AudioInfo(BaseModel):
    duration: float
    sample_rate: int
    channels: int
    format: str
    file_size: int

class AudioProcessor:
    """
    Audio processing utilities for the TTS service.
    """
    
    def __init__(self):
        """Initialize audio processor."""
        logger.info("Initialized Audio Processor")
    
    def get_audio_info(self, file_path: str) -> Optional[AudioInfo]:
        """
        Get information about an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioInfo object or None if error
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file not found: {file_path}")
                return None
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Load audio with librosa
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Detect format from extension
            file_format = os.path.splitext(file_path)[1].lower().replace('.', '')
            
            # Detect channels
            if y.ndim == 1:
                channels = 1
            else:
                channels = y.shape[0]
            
            return AudioInfo(
                duration=duration,
                sample_rate=sr,
                channels=channels,
                format=file_format,
                file_size=file_size
            )
            
        except Exception as e:
            logger.error(f"Error getting audio info for {file_path}: {e}")
            return None
    
    def convert_audio_format(self, input_path: str, output_path: str, 
                           target_format: str = "wav", target_sample_rate: int = None) -> bool:
        """
        Convert audio file to different format.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_format: Target format (wav, mp3, flac, etc.)
            target_sample_rate: Target sample rate (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False
            
            # Load audio
            audio = AudioSegment.from_file(input_path)
            
            # Resample if needed
            if target_sample_rate and audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
            
            # Export in target format
            audio.export(output_path, format=target_format)
            
            logger.info(f"Converted {input_path} to {output_path} ({target_format})")
            return True
            
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return False
    
    def normalize_audio(self, input_path: str, output_path: str = None, 
                       target_lufs: float = -23.0) -> bool:
        """
        Normalize audio to target loudness.
        
        Args:
            input_path: Input audio file path
            output_path: Output path (if None, overwrites input)
            target_lufs: Target loudness in LUFS
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False
            
            output_path = output_path or input_path
            
            # Load audio with librosa
            y, sr = librosa.load(input_path, sr=None)
            
            # Simple RMS-based normalization (basic implementation)
            rms = np.sqrt(np.mean(y**2))
            if rms > 0:
                # Target RMS for approximately -23 LUFS
                target_rms = 0.1  # Approximate value
                scaling_factor = target_rms / rms
                y_normalized = y * scaling_factor
                
                # Prevent clipping
                if np.max(np.abs(y_normalized)) > 1.0:
                    y_normalized = y_normalized / np.max(np.abs(y_normalized)) * 0.95
            else:
                y_normalized = y
            
            # Save normalized audio
            sf.write(output_path, y_normalized, sr)
            
            logger.info(f"Normalized audio: {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {e}")
            return False
    
    def trim_silence(self, input_path: str, output_path: str = None, 
                     silence_threshold: float = 0.01) -> bool:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            input_path: Input audio file path
            output_path: Output path (if None, overwrites input)
            silence_threshold: Threshold for silence detection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False
            
            output_path = output_path or input_path
            
            # Load audio
            y, sr = librosa.load(input_path, sr=None)
            
            # Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Save trimmed audio
            sf.write(output_path, y_trimmed, sr)
            
            logger.info(f"Trimmed silence from: {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return False
    
    def concatenate_audio_files(self, file_paths: list, output_path: str, 
                               crossfade_ms: int = 0) -> bool:
        """
        Concatenate multiple audio files.
        
        Args:
            file_paths: List of audio file paths to concatenate
            output_path: Output file path
            crossfade_ms: Crossfade duration in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not file_paths:
                logger.error("No input files provided")
                return False
            
            # Load first file
            combined = AudioSegment.from_file(file_paths[0])
            
            # Add remaining files
            for file_path in file_paths[1:]:
                if os.path.exists(file_path):
                    audio = AudioSegment.from_file(file_path)
                    if crossfade_ms > 0:
                        combined = combined.append(audio, crossfade=crossfade_ms)
                    else:
                        combined = combined + audio
                else:
                    logger.warning(f"File not found, skipping: {file_path}")
            
            # Export combined audio
            combined.export(output_path, format="wav")
            
            logger.info(f"Concatenated {len(file_paths)} files to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error concatenating audio files: {e}")
            return False
    
    def extract_audio_segment(self, input_path: str, output_path: str,
                             start_time: float, duration: float) -> bool:
        """
        Extract a segment from audio file.
        
        Args:
            input_path: Input audio file path
            output_path: Output file path
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False
            
            # Load audio
            audio = AudioSegment.from_file(input_path)
            
            # Convert to milliseconds
            start_ms = int(start_time * 1000)
            duration_ms = int(duration * 1000)
            
            # Extract segment
            segment = audio[start_ms:start_ms + duration_ms]
            
            # Export segment
            segment.export(output_path, format="wav")
            
            logger.info(f"Extracted segment from {input_path}: {start_time}s-{start_time + duration}s")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting audio segment: {e}")
            return False
    
    def cleanup_old_files(self, directory: str, max_age_hours: int = 24) -> int:
        """
        Clean up old audio files from directory.
        
        Args:
            directory: Directory to clean
            max_age_hours: Maximum age in hours before file is deleted
            
        Returns:
            Number of files deleted
        """
        try:
            if not os.path.exists(directory):
                return 0
            
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            deleted_count = 0
            
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            logger.info(f"Deleted old file: {filename}")
                        except Exception as e:
                            logger.error(f"Error deleting file {filename}: {e}")
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

# Utility functions for audio validation
def is_audio_file(file_path: str) -> bool:
    """Check if file is a supported audio format."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    return os.path.splitext(file_path)[1].lower() in audio_extensions

def get_audio_duration(file_path: str) -> Optional[float]:
    """Get duration of audio file in seconds."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {e}")
        return None

def validate_audio_file(file_path: str, max_duration: float = 300.0, 
                       min_duration: float = 0.1) -> Tuple[bool, str]:
    """
    Validate audio file.
    
    Args:
        file_path: Path to audio file
        max_duration: Maximum allowed duration in seconds
        min_duration: Minimum allowed duration in seconds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        if not is_audio_file(file_path):
            return False, "File is not a supported audio format"
        
        duration = get_audio_duration(file_path)
        if duration is None:
            return False, "Could not read audio file"
        
        if duration < min_duration:
            return False, f"Audio too short: {duration:.2f}s (minimum: {min_duration}s)"
        
        if duration > max_duration:
            return False, f"Audio too long: {duration:.2f}s (maximum: {max_duration}s)"
        
        return True, "Valid audio file"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"