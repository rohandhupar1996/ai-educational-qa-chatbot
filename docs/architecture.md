# AI Education Q&A Bot - Architecture with Voice Cloning

## System Overview

The AI Education Q&A Bot combines OpenAI LLM with real-time voice cloning using SV2TTS (Speaker Verification to Text-To-Speech) to provide intelligent, personalized audio responses.

## Core Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Browser]
        API[API Client]
        REC[Recording Interface]
    end
    
    subgraph "FastAPI Application"
        MAIN[main.py - API Endpoints]
        LLM[llm_model.py - OpenAI]
        TTS_TRAD[tts_traditional.py - gTTS]
        TTS_ADV[tts_advanced.py - Advanced Models]
    end
    
    subgraph "Voice Cloning Pipeline"
        ENC[Encoder - Speaker Verification]
        SYN[Synthesizer - Tacotron2]
        VOC[Vocoder - WaveRNN]
        EMB[Voice Embeddings Store]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        STORAGE[File Storage]
    end
    
    WEB --> MAIN
    API --> MAIN
    REC --> MAIN
    
    MAIN --> LLM
    MAIN --> TTS_TRAD
    MAIN --> ENC
    
    LLM --> OPENAI
    ENC --> EMB
    ENC --> SYN
    SYN --> VOC
    VOC --> STORAGE
```

## Voice Cloning Workflow

```mermaid
sequenceDiagram
    participant User
    participant WebUI
    participant API
    participant Encoder
    participant OpenAI
    participant Synthesizer
    participant Vocoder
    participant Storage
    
    Note over User,Storage: Voice Recording Phase
    User->>WebUI: Record/Upload Audio
    WebUI->>API: POST /record-voice
    API->>Encoder: Process Audio
    Encoder->>Encoder: Create Voice Embedding
    Encoder-->>API: Voice Embedding
    API-->>WebUI: Voice Saved
    
    Note over User,Storage: Voice Cloning Phase
    User->>API: POST /generate-voice-clone
    API->>OpenAI: Generate Text Response
    OpenAI-->>API: AI Answer
    API->>API: Split Text into Chunks
    loop For each chunk
        API->>Synthesizer: Text + Voice Embedding
        Synthesizer-->>API: Mel Spectrogram
        API->>Vocoder: Convert to Audio
        Vocoder-->>API: Audio Chunk
    end
    API->>API: Concatenate Audio
    API->>Storage: Save Final Audio
    Storage-->>API: Audio File Path
    API-->>User: Cloned Voice Response
```

## Voice Cloning Components

### 1. Encoder (Speaker Verification)
```mermaid
graph LR
    AUDIO[Raw Audio] --> PREPROCESS[Preprocessing]
    PREPROCESS --> MEL[Mel Spectrogram]
    MEL --> LSTM[LSTM Network]
    LSTM --> EMBED[256D Embedding]
    EMBED --> STORE[Memory Storage]
```

**Purpose**: Convert voice samples into speaker embeddings
- **Input**: Raw audio (WAV, MP3, FLAC)
- **Output**: 256-dimensional speaker embedding
- **Model**: GE2E (Generalized End-to-End) loss trained LSTM

### 2. Synthesizer (Text-to-Speech)
```mermaid
graph LR
    TEXT[Input Text] --> CHAR[Character Sequence]
    EMBED[Speaker Embedding] --> COND[Conditioning]
    CHAR --> ENC[Encoder]
    ENC --> ATT[Attention]
    COND --> ATT
    ATT --> DEC[Decoder]
    DEC --> MEL[Mel Spectrogram]
```

**Purpose**: Generate mel spectrograms from text and speaker embedding
- **Input**: Text + Speaker embedding
- **Output**: Mel spectrogram
- **Model**: Tacotron2 with speaker conditioning

### 3. Vocoder (Audio Synthesis)
```mermaid
graph LR
    MEL[Mel Spectrogram] --> UPSAMPLE[Upsampling]
    UPSAMPLE --> RNN[WaveRNN]
    RNN --> AUDIO[Raw Audio]
```

**Purpose**: Convert mel spectrograms to audio waveforms
- **Input**: Mel spectrogram
- **Output**: Audio waveform
- **Model**: WaveRNN for high-quality audio generation

## Request Workflows

### 1. Voice Recording Flow
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Encoder
    participant Storage
    
    Client->>API: POST /record-voice {audio_file, voice_id}
    API->>API: Validate Audio Format
    API->>Storage: Save Audio File
    API->>Encoder: Preprocess Audio
    Encoder->>Encoder: Extract Features
    Encoder->>Encoder: Generate Embedding
    Encoder-->>API: 256D Voice Embedding
    API->>API: Store in Memory
    API-->>Client: {voice_id, status, embedding_size}
```

### 2. Voice Cloning Generation Flow
```mermaid
sequenceDiagram
    participant Client
    participant API
    participant OpenAI
    participant Synthesizer
    participant Vocoder
    participant Storage
    
    Client->>API: POST /generate-voice-clone
    API->>API: Validate Voice ID Exists
    API->>OpenAI: Generate Response
    OpenAI-->>API: AI Answer Text
    API->>API: Split Text (3-word chunks)
    loop For each chunk
        API->>Synthesizer: Text + Voice Embedding
        Synthesizer-->>API: Mel Spectrogram
        API->>Vocoder: Generate Audio
        Vocoder-->>API: Audio Segment
    end
    API->>API: Concatenate Segments
    API->>Storage: Save Final Audio
    API-->>Client: {success, audio_file, answer}
```

### 3. Browser Recording Interface
```mermaid
sequenceDiagram
    participant Browser
    participant Microphone
    participant WebAPI
    participant Backend
    
    Browser->>Microphone: navigator.mediaDevices.getUserMedia()
    Microphone-->>Browser: Audio Stream
    Browser->>Browser: MediaRecorder.start()
    Browser->>Browser: Record Audio Chunks
    Browser->>Browser: Create Audio Blob
    Browser->>WebAPI: FormData with audio
    WebAPI->>Backend: POST /record-voice
    Backend-->>Browser: Voice Saved Response
```

## Data Flow Architecture

```mermaid
graph TD
    REQUEST[HTTP Request] --> VALIDATE[Input Validation]
    VALIDATE --> ROUTE[Route Handler]
    
    subgraph "Voice Processing"
        ROUTE --> VOICE_CHECK{Voice ID Exists?}
        VOICE_CHECK -->|No| RECORD[Record Voice]
        VOICE_CHECK -->|Yes| GENERATE[Generate Response]
        RECORD --> ENCODER[Encoder Processing]
        ENCODER --> EMBEDDING[Store Embedding]
    end
    
    subgraph "AI Processing"
        GENERATE --> OPENAI[OpenAI API]
        OPENAI --> SPLIT[Split Long Text]
    end
    
    subgraph "Audio Synthesis"
        SPLIT --> SYNTHESIZER[Tacotron2]
        SYNTHESIZER --> VOCODER[WaveRNN]
        VOCODER --> CONCAT[Concatenate Audio]
    end
    
    CONCAT --> RESPONSE[HTTP Response]
    EMBEDDING --> RESPONSE
```

## File Structure with Voice Cloning

```
src/
├── scripts/
│   ├── main.py                 # FastAPI app with voice endpoints
│   ├── llm_model.py           # OpenAI integration
│   ├── tts_traditional.py     # Traditional TTS
│   └── tts_advanced.py        # Advanced TTS models
├── encoder/                   # SV2TTS Encoder
│   ├── inference.py          # Speaker embedding extraction
│   ├── model.py              # LSTM speaker encoder
│   └── audio.py              # Audio preprocessing
├── synthesizer/               # Tacotron2 Synthesizer
│   ├── inference.py          # Text-to-mel conversion
│   ├── models/               # Tacotron2 model
│   └── utils/                # Text processing utilities
├── vocoder/                   # WaveRNN Vocoder
│   ├── inference.py          # Mel-to-audio conversion
│   └── models/               # WaveRNN model
├── saved_models/              # Pre-trained models
│   └── default/
│       ├── encoder.pt        # Speaker encoder weights
│       ├── synthesizer.pt    # Tacotron2 weights
│       └── vocoder.pt        # WaveRNN weights
├── voices/                    # Stored voice files
└── outputs/                   # Generated audio files
```

## API Endpoints

| Endpoint | Method | Purpose | Input | Output |
|----------|---------|---------|-------|--------|
| `/record-interface` | GET | Recording UI | - | HTML Interface |
| `/record-voice` | POST | Save voice | Audio file + voice_id | Voice embedding |
| `/generate-voice-clone` | POST | Clone voice | Text + voice_id | Cloned audio |
| `/voices` | GET | List voices | - | Available voice IDs |
| `/ask` | POST | Standard Q&A | Question | Text response |
| `/download/{file}` | GET | Get audio | Filename | Audio file |

## Memory Management

### Voice Embeddings Storage
```python
# In-memory storage
voice_embeddings: Dict[str, np.ndarray] = {}

# Persistent storage (future enhancement)
def save_embeddings():
    with open('voice_embeddings.pkl', 'wb') as f:
        pickle.dump(voice_embeddings, f)
```

### Audio Processing Pipeline
- **Input**: Various audio formats (MP3, WAV, FLAC)
- **Preprocessing**: 16kHz sampling, noise reduction
- **Chunking**: 3-word segments for quality
- **Output**: High-quality WAV files

## Performance Optimizations

### Text Chunking Strategy
```python
def split_text_for_voice(text, max_words=3):
    # Split into small chunks for better voice quality
    # Prevents attention drift in long sequences
```

### Model Loading
- **Lazy Loading**: Models loaded on first use
- **GPU Support**: Automatic GPU detection
- **Memory Efficient**: Models shared across requests

## Deployment Architecture

### Docker Configuration
```yaml
services:
  ai-voice-bot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./voices:/app/voices      # Voice files
      - ./outputs:/app/outputs    # Generated audio
      - ./saved_models:/app/saved_models  # Model weights
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

### Scaling Considerations
- **Stateless Design**: Voice embeddings in shared storage
- **Load Balancing**: Multiple FastAPI instances
- **Model Caching**: Shared model weights across containers

## Security & Privacy

### Voice Data Protection
- **Temporary Storage**: Voice files can be auto-deleted
- **Embedding Only**: Store embeddings, not raw audio
- **Access Control**: Voice ID-based permissions

### API Security
- **CORS Configuration**: Controlled origin access
- **Rate Limiting**: Prevent abuse
- **Input Validation**: Audio format verification