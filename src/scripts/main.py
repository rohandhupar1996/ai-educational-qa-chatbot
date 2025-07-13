"""
AI Education Q&A Bot - Main FastAPI Service with Voice Cloning
"""
import os
import logging
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import soundfile as sf
from fastapi.responses import HTMLResponse

import numpy as np
from contextlib import asynccontextmanager

# Import existing modules
from llm_model import OpenAIModel, QuestionRequest, QuestionResponse
from tts_traditional import GoogleTTS, TTSRequest, TTSResponse

# Import voice cloning modules
import sys
sys.path.append('src')

from  encoder import inference as encoder
from  synthesizer.inference import Synthesizer
from  vocoder import inference as vocoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
openai_model = None
google_tts = None
synthesizer = None
voice_embeddings: Dict[str, np.ndarray] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global openai_model, google_tts, synthesizer
    
    # Startup
    logger.info("Starting AI Education Q&A Bot with Voice Cloning...")
    try:
        # Initialize existing services
        openai_model = OpenAIModel()
        google_tts = GoogleTTS()
        
        # Initialize voice cloning models
        logger.info("Loading voice cloning models...")
        encoder.load_model("/Users/rohan/Downloads/ai-educational-qa-chatbot/src/saved_models/default/encoder.pt")
        synthesizer = Synthesizer("/Users/rohan/Downloads/ai-educational-qa-chatbot/src/saved_models/default/synthesizer.pt")
        vocoder.load_model("/Users/rohan/Downloads/ai-educational-qa-chatbot/src/saved_models/default/vocoder.pt")
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Education Q&A Bot...")

# Create FastAPI app
app = FastAPI(
    title="AI Education Q&A Bot with Voice Cloning",
    description="An intelligent Q&A bot with voice cloning capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class VoiceCloneRequest(BaseModel):
    question: str
    voice_id: str
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7

class VoiceCloneResponse(BaseModel):
    success: bool
    question: str
    answer: str
    audio_file: Optional[str] = None
    voice_id: str
    ai_model_used: str
    error: Optional[str] = None

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("voices", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "AI Education Q&A Bot with Voice Cloning",
        "version": "1.0.0",
        "status": "running",
        "ai_provider": "OpenAI",
        "voice_cloning": "SV2TTS",
        "endpoints": {
            "record_voice": "/record-voice",
            "generate_voice_clone": "/generate-voice-clone",
            "ask_question": "/ask",
            "list_voices": "/voices",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_initialized": openai_model is not None,
        "synthesizer_initialized": synthesizer is not None,
        "voice_cloning": "enabled"
    }

@app.post("/record-voice")
async def record_voice(voice_id: str, audio_file: UploadFile = File(...)):
    """Record and store a voice sample for cloning."""
    try:
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(400, detail="File must be audio format")
        
        # Save uploaded audio
        audio_path = f"voices/{voice_id}.wav"
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Preprocess and create voice embedding
        wav = encoder.preprocess_wav(audio_path)
        embed = encoder.embed_utterance(wav)
        
        # Store embedding
        voice_embeddings[voice_id] = embed
        
        logger.info(f"Voice recorded and processed for ID: {voice_id}")
        
        return {
            "voice_id": voice_id,
            "status": "recorded",
            "embedding_size": embed.shape[0],
            "audio_duration": len(wav) / 16000  # Assuming 16kHz sample rate
        }
        
    except Exception as e:
        logger.error(f"Error recording voice: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/generate-voice-clone", response_model=VoiceCloneResponse)
async def generate_with_voice_clone(request: VoiceCloneRequest):
    """Generate response with voice cloning."""
    try:
        # Check if voice exists
        if request.voice_id not in voice_embeddings:
            return VoiceCloneResponse(
                success=False,
                question=request.question,
                answer="",
                voice_id=request.voice_id,
                ai_model_used=request.model_name,
                error=f"Voice ID '{request.voice_id}' not found. Please record voice first."
            )
        
        # Generate OpenAI response
        if request.model_name != openai_model.model_name:
            temp_model = OpenAIModel(model_name=request.model_name)
            answer_response = temp_model.generate_answer(request.question)
        else:
            answer_response = openai_model.generate_answer(request.question)
        
        if not answer_response.success:
            return VoiceCloneResponse(
                success=False,
                question=request.question,
                answer="",
                voice_id=request.voice_id,
                ai_model_used=request.model_name,
                error=f"Failed to generate answer: {answer_response.error}"
            )
        
        # Get voice embedding
        embed = voice_embeddings[request.voice_id]

        # Generate speech with cloned voice
        def split_text_for_voice(text, max_words=15):
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), max_words):
                chunk = ' '.join(words[i:i + max_words])
                chunks.append(chunk)
            
            return chunks
        # Process text in chunks for better quality
        text_chunks = split_text_for_voice(answer_response.answer)
        wav_segments = []

        for chunk in text_chunks:
            specs = synthesizer.synthesize_spectrograms([chunk], [embed])
            wav_segment = vocoder.infer_waveform(specs[0])
            wav_segments.append(wav_segment)

        # Concatenate all segments
        wav = np.concatenate(wav_segments)

        # Save audio file
        audio_filename = f"cloned_{request.voice_id}_{hash(request.question) % 10000}.wav"
        audio_path = f"outputs/{audio_filename}"
        sf.write(audio_path, wav, synthesizer.sample_rate)

        logger.info(f"Generated voice clone response for voice ID: {request.voice_id}")
        
        return VoiceCloneResponse(
            success=True,
            question=request.question,
            answer=answer_response.answer,
            audio_file=audio_filename,
            voice_id=request.voice_id,
            ai_model_used=request.model_name
        )
        
    except Exception as e:
        logger.error(f"Error in voice clone generation: {e}")
        return VoiceCloneResponse(
            success=False,
            question=request.question,
            answer="",
            voice_id=request.voice_id,
            ai_model_used=request.model_name,
            error=str(e)
        )

@app.get("/voices")
async def list_voices():
    """List available voice IDs."""
    return {
        "voices": list(voice_embeddings.keys()),
        "count": len(voice_embeddings)
    }

@app.delete("/voice/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice profile."""
    if voice_id not in voice_embeddings:
        raise HTTPException(404, detail="Voice not found")
    
    # Remove from memory
    del voice_embeddings[voice_id]
    
    # Remove audio file if exists
    audio_path = f"voices/{voice_id}.wav"
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    return {"voice_id": voice_id, "status": "deleted"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question without voice cloning (original functionality)."""
    try:
        if not openai_model:
            raise HTTPException(500, detail="OpenAI model not initialized")
        
        response = openai_model.generate_answer(
            question=request.question,
            max_lines=request.max_length,
            context=request.context
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Google TTS (original functionality)."""
    try:
        if not google_tts:
            raise HTTPException(500, detail="TTS service not initialized")
        
        output_file = f"outputs/tts_{hash(request.text) % 10000}_{request.language}.mp3"
        
        response = google_tts.synthesize(
            text=request.text,
            lang=request.language,
            slow=request.slow,
            output_file=output_file
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        raise HTTPException(500, detail=str(e))

@app.get("/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file."""
    file_path = f"outputs/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Audio file not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )
# Add this endpoint anywhere in your main.py (keep everything else as-is)
@app.get("/record-interface", response_class=HTMLResponse)
async def recording_interface():
    """Voice recording and upload interface for voice cloning"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Recording</title>
        <style>
            .container { max-width: 700px; margin: 30px auto; padding: 20px; font-family: Arial, sans-serif; }
            .section { border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 6px; }
            button { padding: 8px 16px; margin: 5px; border: none; border-radius: 4px; cursor: pointer; }
            .record-btn { background: #4CAF50; color: white; }
            .recording { background: #f44336; color: white; }
            .save-btn { background: #2196F3; color: white; }
            input, textarea { padding: 6px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
            audio { width: 100%; margin: 8px 0; }
            .status { margin: 8px 0; font-weight: bold; }
            .success { color: green; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🎙️ Voice Recording for Cloning</h2>
            
            <div class="section">
                <h3>Record Voice</h3>
                <button id="recordBtn" class="record-btn">Start Recording</button>
                <button id="stopBtn" disabled>Stop</button>
                <button id="playBtn" disabled>Play</button>
                <audio id="recordedAudio" controls style="display:none;"></audio>
                <div class="status" id="recordStatus">Ready to record</div>
            </div>

            <div class="section">
                <h3>Or Upload Audio File</h3>
                <input type="file" id="fileInput" accept="audio/*" />
                <audio id="uploadedAudio" controls style="display:none;"></audio>
                <div class="status" id="uploadStatus">Select audio file</div>
            </div>

            <div class="section">
                <h3>Save Voice</h3>
                <input type="text" id="voiceId" placeholder="Voice ID (e.g. 'my_voice')" style="width: 200px;" />
                <button id="saveBtn" disabled class="save-btn">Save Voice</button>
                <div class="status" id="saveStatus"></div>
            </div>

            <div class="section">
                <h3>Test Voice Cloning</h3>
                <textarea id="testText" placeholder="Enter text to test..." style="width: 400px; height: 60px;">Hello, this is a test of voice cloning.</textarea><br>
                <select id="voiceSelect" style="width: 150px;">
                    <option value="">Select voice...</option>
                </select>
                <button id="testBtn" class="save-btn">Generate Speech</button>
                <audio id="testAudio" controls style="display:none;"></audio>
                <div class="status" id="testStatus"></div>
            </div>
        </div>

        <script>
            let mediaRecorder, audioChunks = [], currentBlob = null;

            // Initialize microphone
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = () => {
                        currentBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        document.getElementById('recordedAudio').src = URL.createObjectURL(currentBlob);
                        document.getElementById('recordedAudio').style.display = 'block';
                        document.getElementById('playBtn').disabled = false;
                        document.getElementById('saveBtn').disabled = false;
                        document.getElementById('recordStatus').textContent = 'Recording complete';
                    };
                })
                .catch(() => {
                    document.getElementById('recordStatus').textContent = 'Microphone access denied';
                    document.getElementById('recordStatus').className = 'status error';
                });

            // Recording controls
            document.getElementById('recordBtn').onclick = () => {
                audioChunks = [];
                mediaRecorder.start();
                document.getElementById('recordBtn').disabled = true;
                document.getElementById('recordBtn').className = 'recording';
                document.getElementById('recordBtn').textContent = 'Recording...';
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('recordStatus').textContent = 'Recording...';
            };

            document.getElementById('stopBtn').onclick = () => {
                mediaRecorder.stop();
                document.getElementById('recordBtn').disabled = false;
                document.getElementById('recordBtn').className = 'record-btn';
                document.getElementById('recordBtn').textContent = 'Start Recording';
                document.getElementById('stopBtn').disabled = true;
            };

            document.getElementById('playBtn').onclick = () => {
                document.getElementById('recordedAudio').play();
            };

            // File upload
            document.getElementById('fileInput').onchange = (e) => {
                const file = e.target.files[0];
                if (file) {
                    currentBlob = file;
                    document.getElementById('uploadedAudio').src = URL.createObjectURL(file);
                    document.getElementById('uploadedAudio').style.display = 'block';
                    document.getElementById('saveBtn').disabled = false;
                    document.getElementById('uploadStatus').textContent = `File: ${file.name}`;
                    document.getElementById('uploadStatus').className = 'status success';
                }
            };

            // Save voice
            document.getElementById('saveBtn').onclick = async () => {
                const voiceId = document.getElementById('voiceId').value.trim();
                if (!voiceId || !currentBlob) {
                    alert('Enter voice ID and record/upload audio');
                    return;
                }

                const formData = new FormData();
                formData.append('audio_file', currentBlob, `${voiceId}.wav`);

                try {
                    document.getElementById('saveStatus').textContent = 'Saving...';
                    const response = await fetch(`/record-voice?voice_id=${voiceId}`, {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    if (response.ok) {
                        document.getElementById('saveStatus').textContent = `✅ Saved: ${data.voice_id}`;
                        document.getElementById('saveStatus').className = 'status success';
                        loadVoices();
                    } else {
                        throw new Error(data.detail);
                    }
                } catch (error) {
                    document.getElementById('saveStatus').textContent = `❌ Error: ${error.message}`;
                    document.getElementById('saveStatus').className = 'status error';
                }
            };

            // Load available voices
            async function loadVoices() {
                try {
                    const response = await fetch('/voices');
                    const data = await response.json();
                    const select = document.getElementById('voiceSelect');
                    select.innerHTML = '<option value="">Select voice...</option>';
                    data.voices.forEach(voice => {
                        select.innerHTML += `<option value="${voice}">${voice}</option>`;
                    });
                } catch (error) {
                    console.error('Failed to load voices:', error);
                }
            }

            // Test voice cloning
            document.getElementById('testBtn').onclick = async () => {
                const text = document.getElementById('testText').value.trim();
                const voiceId = document.getElementById('voiceSelect').value;
                
                if (!text || !voiceId) {
                    alert('Enter text and select voice');
                    return;
                }

                try {
                    document.getElementById('testStatus').textContent = 'Generating...';
                    const response = await fetch('/generate-voice-clone', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            question: text,
                            voice_id: voiceId
                        })
                    });

                    const data = await response.json();
                    if (data.success) {
                        document.getElementById('testAudio').src = `/download/${data.audio_file}`;
                        document.getElementById('testAudio').style.display = 'block';
                        document.getElementById('testStatus').textContent = '✅ Generated!';
                        document.getElementById('testStatus').className = 'status success';
                    } else {
                        throw new Error(data.error);
                    }
                } catch (error) {
                    document.getElementById('testStatus').textContent = `❌ Error: ${error.message}`;
                    document.getElementById('testStatus').className = 'status error';
                }
            };

            // Initialize
            loadVoices();
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )