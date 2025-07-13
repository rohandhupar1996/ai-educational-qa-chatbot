#!/usr/bin/env python3
"""
Batch conversion script for processing multiple text files to speech using OpenAI.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.scripts.llm_model import OpenAIModel
from src.scripts.tts_traditional import GoogleTTS
from src.scripts.tts_advanced import TacotronTTS, OuteTTS, GlowTTS, YourTTS
from src.scripts.audio_utils import AudioProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Batch processor for converting text files to speech using various TTS models with OpenAI.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize batch processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.openai_model = None
        self.tts_models = {}
        self.audio_processor = AudioProcessor()
        self.results = []
        
        # Initialize models based on config
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI and TTS models."""
        try:
            # Initialize OpenAI if enabled
            if self.config.get('use_openai', False):
                model_name = self.config.get('openai_model', 'gpt-3.5-turbo')
                self.openai_model = OpenAIModel(model_name=model_name)
                logger.info(f"Initialized OpenAI model: {model_name}")
            
            # Initialize TTS models
            models_to_init = self.config.get('tts_models', ['gtts'])
            
            for model_name in models_to_init:
                if model_name == 'gtts':
                    self.tts_models['gtts'] = GoogleTTS()
                elif model_name == 'tacotron':
                    self.tts_models['tacotron'] = TacotronTTS()
                elif model_name == 'outetts':
                    self.tts_models['outetts'] = OuteTTS()
                elif model_name == 'glow-tts':
                    self.tts_models['glow-tts'] = GlowTTS()
                elif model_name == 'yourtts':
                    self.tts_models['yourtts'] = YourTTS()
                else:
                    logger.warning(f"Unknown TTS model: {model_name}")
            
            logger.info(f"Initialized TTS models: {list(self.tts_models.keys())}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def process_text_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a single text file.
        
        Args:
            file_path: Path to input text file
            output_dir: Output directory for audio files
            
        Returns:
            Processing result dictionary
        """
        try:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
            
            if not text_content:
                return {
                    'file': file_path,
                    'success': False,
                    'error': 'Empty text file',
                    'audio_files': []
                }
            
            # Generate answer using OpenAI if enabled
            if self.openai_model and self.config.get('use_openai', False):
                response = self.openai_model.generate_answer(text_content)
                if response.success:
                    text_to_convert = response.answer
                else:
                    text_to_convert = text_content
                    logger.warning(f"OpenAI generation failed for {file_path}: {response.error}")
            else:
                text_to_convert = text_content
            
            # Generate audio with each TTS model
            file_stem = Path(file_path).stem
            audio_files = []
            errors = []
            
            for model_name, model in self.tts_models.items():
                try:
                    output_filename = f"{file_stem}_{model_name}.wav"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    if model_name == 'gtts':
                        response = model.synthesize(
                            text=text_to_convert,
                            lang=self.config.get('language', 'en'),
                            output_file=output_path.replace('.wav', '.mp3')
                        )
                        if response.success:
                            audio_files.append(response.output_file)
                    
                    elif model_name == 'tacotron':
                        response = model.synthesize(
                            text=text_to_convert,
                            output_file=output_path,
                            speed=self.config.get('tacotron_speed', 1.2)
                        )
                        if response.success:
                            audio_files.append(response.output_file)
                    
                    elif model_name == 'outetts':
                        response = model.synthesize(
                            text=text_to_convert,
                            output_file=output_path,
                            temperature=self.config.get('outetts_temperature', 0.1),
                            speaker_name=self.config.get('outetts_speaker', 'male_2')
                        )
                        if response.success:
                            audio_files.append(response.output_file)
                    
                    elif model_name == 'glow-tts':
                        response = model.synthesize(
                            text=text_to_convert,
                            output_file=output_path,
                            sample_rate=self.config.get('sample_rate', 22050)
                        )
                        if response.success:
                            audio_files.append(response.output_file)
                    
                    elif model_name == 'yourtts':
                        response = model.synthesize(
                            text=text_to_convert,
                            output_file=output_path,
                            speaker_index=self.config.get('yourtts_speaker_index', 0),
                            language_index=self.config.get('yourtts_language_index', 0)
                        )
                        if response.success:
                            audio_files.append(response.output_file)
                    
                    logger.info(f"Generated audio with {model_name} for {file_path}")
                    
                except Exception as e:
                    error_msg = f"Error with {model_name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Post-process audio files if enabled
            if self.config.get('normalize_audio', False):
                for audio_file in audio_files:
                    self.audio_processor.normalize_audio(audio_file)
            
            if self.config.get('trim_silence', False):
                for audio_file in audio_files:
                    self.audio_processor.trim_silence(audio_file)
            
            return {
                'file': file_path,
                'success': len(audio_files) > 0,
                'audio_files': audio_files,
                'errors': errors,
                'processing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                'file': file_path,
                'success': False,
                'error': str(e),
                'audio_files': []
            }
    
    def process_batch(self, input_files: List[str], output_dir: str, 
                     max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Process multiple files in parallel.
        
        Args:
            input_files: List of input file paths
            output_dir: Output directory
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of processing results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_text_file, file_path, output_dir): file_path
                for file_path in input_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        logger.info(f"✓ Processed: {file_path}")
                    else:
                        logger.error(f"✗ Failed: {file_path} - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"✗ Exception processing {file_path}: {e}")