# AI Education Q&A Bot 🤖🎵 (OpenAI Powered with Voice Cloning)

An intelligent Q&A bot that combines **OpenAI GPT models** with **real-time voice cloning** to generate spoken responses in any voice. Perfect for creating personalized educational content or building voice-enabled AI assistants.

## ✨ Features

- **🧠 AI-Powered Q&A**: OpenAI GPT models generate intelligent responses 
- **🎭 Real-Time Voice Cloning**: Clone any voice and speak AI responses
- **🎤 Live Recording**: Browser-based microphone recording
- **📁 File Upload**: Support for MP3, WAV, FLAC audio formats
- **🗣️ Multiple TTS Models**: gTTS, Tacotron2, OuteTTS, Glow-TTS, YourTTS
- **🚀 FastAPI Service**: RESTful API with auto-documentation
- **🐳 Docker Ready**: One-command deployment
- **🔊 Audio Processing**: Built-in audio enhancement and processing
- **📱 Multi-format Support**: Generate MP3, WAV audio outputs
- **🎯 Model Selection**: Choose from multiple OpenAI models (GPT-3.5, GPT-4, etc.)

## 🎭 Voice Cloning Workflow

1. **Record/Upload Voice**: Use browser recording or upload audio files
2. **Save Voice Profile**: Create embeddings using SV2TTS encoder
3. **Generate AI Response**: Get intelligent answers from OpenAI
4. **Clone Voice**: Speak the response in the cloned voice
5. **Download Audio**: Get the final voice-cloned response

## 🚀 Quick Start

### 1. Setup
```bash
git clone <repo-url>
cd ai-edu-qa-bot
cp .env.example .env
# Add your OPENAI_API_KEY to .env
pip install -r requirements.txt
```

### 2. Download Voice Models
```bash
# Models will auto-download on first run or manually:
mkdir -p saved_models/default
# Download from: https://github.com/CorentinJ/Real-Time-Voice-Cloning
```

### 3. Run (Development)
```bash
python src/scripts/main.py
```

### 4. Use Voice Recording Interface
Visit: `http://localhost:8000/record-interface`

## 📖 API Usage

### Record Voice for Cloning
```bash
curl -X POST "http://localhost:8000/record-voice?voice_id=my_voice" \
     -F "audio_file=@voice.wav"
```

### Generate Voice-Cloned Response
```bash
curl -X POST "http://localhost:8000/generate-voice-clone" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is machine learning?",
       "voice_id": "my_voice",
       "model_name": "gpt-3.5-turbo"
     }'
```

### Standard Q&A (No Voice)
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Explain neural networks"}'
```

## 🎤 Voice Recording Options

### Browser Recording
- Live microphone recording
- Real-time preview
- Automatic WAV conversion

### File Upload
- Support for MP3, WAV, FLAC, M4A
- Drag & drop interface
- Audio preview before saving

### Voice Management
- List all saved voices: `GET /voices`
- Delete voice profiles: `DELETE /voice/{voice_id}`

## 🤖 Available AI Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **gpt-3.5-turbo** | Fast, cost-effective | Quick Q&A, general education |
| **gpt-4** | High quality, detailed | Complex explanations |
| **gpt-4-turbo** | Faster GPT-4 | Balanced quality/speed |
| **gpt-4o** | Latest optimized | Best performance |

## 🎭 Voice Cloning Technology

- **Encoder**: SV2TTS speaker verification model
- **Synthesizer**: Tacotron2 for mel-spectrogram generation
- **Vocoder**: WaveRNN for audio synthesis
- **Quality**: 3-second chunks for optimal voice quality

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_OPENAI_MODEL=gpt-3.5-turbo
USE_GPU=false
DEFAULT_TTS_MODEL=gtts
DEFAULT_LANGUAGE=en
```

### Voice Cloning Parameters
```json
{
  "question": "What is deep learning?",
  "voice_id": "celebrity_voice",
  "model_name": "gpt-4",
  "max_tokens": 500,
  "temperature": 0.7
}
```

## 📁 Project Structure
```
src/
├── scripts/
│   ├── main.py              # FastAPI application with voice cloning
│   ├── llm_model.py         # OpenAI integration
│   ├── tts_traditional.py   # gTTS implementation
│   └── tts_advanced.py      # Neural TTS models
├── encoder/                 # SV2TTS encoder
├── synthesizer/             # Tacotron2 synthesizer
├── vocoder/                 # WaveRNN vocoder
└── saved_models/            # Pre-trained models
```

## 🎯 Example Use Cases

### Educational Content Creation
```python
# Generate lesson audio with voice cloning
response = requests.post("http://localhost:8000/generate-voice-clone", json={
    "question": "Explain convolutional neural networks",
    "voice_id": "teacher_voice",
    "model_name": "gpt-4"
})
```

### Celebrity Voice Responses
```python
# Get AI responses in any voice
response = requests.post("http://localhost:8000/generate-voice-clone", json={
    "question": "What is reinforcement learning?",
    "voice_id": "morgan_freeman_voice"
})
```

## 🐳 Docker Deployment

### Quick Start
```bash
cp .env.example .env
echo "OPENAI_API_KEY=your_actual_api_key_here" >> .env
docker-compose up -d
```

### Production
```bash
docker run -d \
  --name ai-voice-bot \
  -p 80:8000 \
  -e OPENAI_API_KEY=your_key \
  -v /data/voices:/app/voices \
  -v /data/outputs:/app/outputs \
  --restart always \
  ai-edu-qa-bot:latest
```

## 🚀 Performance Tips

- **Voice Quality**: Use 5-30 second clear voice samples
- **Text Length**: Keep responses under 3 words per chunk for best quality
- **GPU**: Enable GPU for faster synthesis: `USE_GPU=true`
- **Caching**: Voice embeddings persist in memory during session

## 🔧 Development

### Adding New Voices
1. Record/upload audio via `/record-interface`
2. Voice embedding automatically created
3. Use `voice_id` in generation requests

### Custom Voice Processing
```python
from encoder import inference as encoder

# Process custom audio
wav = encoder.preprocess_wav("path/to/audio.wav")
embed = encoder.embed_utterance(wav)
```

## 📊 Monitoring

- Health check: `GET /health`
- Available voices: `GET /voices`
- Recording interface: `GET /record-interface`
- API docs: `GET /docs`

## 🎮 Demo

Try these examples:

**Record Your Voice:**
Visit `http://localhost:8000/record-interface`

**API Voice Cloning:**
```bash
curl -X POST localhost:8000/generate-voice-clone -H "Content-Type: application/json" \
  -d '{"question": "Hello, this is a test of voice cloning!", "voice_id": "my_voice"}'
```

## 🤝 Contributing

1. Fork the repo
2. Add voice cloning features
3. Test with various voice samples
4. Submit PR

## 📄 License

MIT License - Build, modify, and share freely!

---

**🎯 Perfect for:** Content Creators, Educators, Voice App Developers, AI Researchers

**💡 New Feature:** Real-time voice cloning with OpenAI responses!