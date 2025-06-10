# 🎥 Video RAG System with Gemini API

A comprehensive Retrieval-Augmented Generation (RAG) system for video content analysis and intelligent querying using Google's Gemini API. This system is specifically designed for CMake tutorial videos but can be adapted for any technical video content.

## 🌟 Features

- **Video Processing**: Automatic transcription, content extraction, and analysis
- **Gemini Integration**: Advanced video analysis using Google's multimodal AI
- **Vector Search**: Semantic search across video content using embeddings
- **Intelligent Q&A**: Natural language queries with context-aware responses
- **Web Interface**: User-friendly web UI for video upload and querying
- **REST API**: Complete API for programmatic access
- **CMake Integration**: Built with CMake for easy integration with existing projects

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│ Video Processor │───▶│  Gemini API     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI/API    │◀───│   RAG Engine    │───▶│ Vector Database │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **Video Processor**: Handles video upload, transcription (Whisper), and metadata extraction
2. **Gemini Client**: Interfaces with Google's Gemini API for video analysis and text generation
3. **Vector Store**: ChromaDB-based storage for embeddings and semantic search
4. **RAG Engine**: Orchestrates retrieval and generation for user queries
5. **API Layer**: FastAPI-based REST API with web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CMake 3.16+
- FFmpeg (for video processing)
- Google Gemini API key

### Installation

1. **Clone and navigate to the project**:
   ```bash
   cd video-rag-gemini
   ```

2. **Build with CMake**:
   ```bash
   mkdir build && cd build
   cmake ..
   make install_python_deps
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

4. **Setup environment**:
   ```bash
   make setup_environment
   ```

5. **Start the server**:
   ```bash
   make run_server
   ```

6. **Access the web interface**:
   Open http://localhost:8000 in your browser

### Manual Installation

If you prefer not to use CMake:

```bash
# Install dependencies
pip install -r requirements.txt

# Setup directories
mkdir -p data/{uploads,processed,chroma_db}

# Configure environment
cp .env.template .env
# Edit .env with your settings

# Start the server
python -m uvicorn src.video_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 Usage

### Web Interface

1. **Upload Videos**: Use the web interface to upload CMake tutorial videos
2. **Process Content**: The system automatically transcribes and analyzes videos
3. **Ask Questions**: Query your video library using natural language
4. **Browse Results**: View responses with source attribution and confidence scores

### API Usage

#### Upload a Video

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@your_video.mp4" \
  -F "title=CMake Basics Tutorial" \
  -F "cmake_topics=build system,targets" \
  -F "difficulty_level=Beginner"
```

#### Query Videos

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a static library in CMake?",
    "max_results": 5,
    "filter_by_topics": ["library", "build"]
  }'
```

#### List Videos

```bash
curl "http://localhost:8000/api/v1/videos"
```

### Python API

```python
from src.video_rag import VideoRAGEngine

# Initialize the engine
rag = VideoRAGEngine()

# Process a video
video_id = await rag.process_video(
    "path/to/video.mp4",
    metadata={
        "title": "CMake Tutorial",
        "cmake_topics": ["basics", "targets"],
        "difficulty_level": "Beginner"
    }
)

# Query the system
result = await rag.query("How to link libraries in CMake?")
print(result.response)
```

## 🔧 Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Gemini API key (required) | - |
| `MAX_VIDEO_SIZE_MB` | Maximum video file size | 100 |
| `GEMINI_MODEL` | Gemini model to use | gemini-1.5-pro |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `API_PORT` | API server port | 8000 |

## 📊 Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)

## 🧪 Testing

Run the test suite:

```bash
make run_tests
# or
python -m pytest tests/ -v
```

## 📚 Examples

### Process a CMake Tutorial Video

```python
# examples/process_cmake_video.py
import asyncio
from src.video_rag import VideoRAGEngine

async def main():
    rag = VideoRAGEngine()
    
    # Process video
    video_id = await rag.process_video(
        "cmake_tutorial.mp4",
        metadata={
            "title": "CMake Build System Basics",
            "description": "Introduction to CMake build system",
            "cmake_topics": ["basics", "build system", "targets"],
            "difficulty_level": "Beginner"
        }
    )
    
    print(f"Video processed: {video_id}")
    
    # Query the video
    result = await rag.query("What is CMake?")
    print(f"Answer: {result.response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Batch Processing

```python
# examples/batch_process.py
import asyncio
import os
from src.video_rag import VideoRAGEngine

async def process_video_directory(directory_path):
    rag = VideoRAGEngine()
    
    for filename in os.listdir(directory_path):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(directory_path, filename)
            
            try:
                video_id = await rag.process_video(video_path)
                print(f"Processed: {filename} -> {video_id}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Usage
asyncio.run(process_video_directory("./videos"))
```

## 🔍 API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🛠️ Development

### Project Structure

```
video-rag-gemini/
├── src/video_rag/           # Main package
│   ├── api/                 # FastAPI application
│   ├── core/                # Core models and exceptions
│   ├── config.py            # Configuration management
│   ├── gemini_client.py     # Gemini API integration
│   ├── video_processor.py   # Video processing pipeline
│   ├── vector_store.py      # Vector database operations
│   ├── embeddings.py        # Embedding generation
│   └── rag_engine.py        # Main RAG engine
├── examples/                # Usage examples
├── tests/                   # Test suite
├── docs/                    # Documentation
├── CMakeLists.txt          # CMake configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Adding New Features

1. **Video Analyzers**: Extend `VideoProcessor` for new content types
2. **Embedding Models**: Customize `EmbeddingGenerator` for domain-specific embeddings
3. **Query Processors**: Add specialized query handling in `RAGEngine`
4. **API Endpoints**: Add new routes in `api/routes.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the cmake-examples repository and follows the same license terms.

## 🆘 Troubleshooting

### Common Issues

1. **Gemini API Key Error**:
   - Ensure your API key is correctly set in `.env`
   - Check API key permissions and quotas

2. **Video Processing Fails**:
   - Verify FFmpeg is installed
   - Check video file format and size limits
   - Ensure sufficient disk space

3. **Embedding Model Download**:
   - First run may take time to download models
   - Ensure internet connection for model downloads

4. **Memory Issues**:
   - Reduce `MAX_CHUNK_SIZE` for large videos
   - Process videos in smaller batches

### Getting Help

- Check the [API documentation](http://localhost:8000/docs)
- Review the examples in the `examples/` directory
- Check system logs for detailed error messages

## 🔮 Future Enhancements

- [ ] Support for live video streaming
- [ ] Multi-language transcription
- [ ] Advanced video analytics (object detection, scene analysis)
- [ ] Integration with popular video platforms
- [ ] Real-time collaborative querying
- [ ] Custom embedding fine-tuning for CMake content

---

Built with ❤️ for the CMake community using Google's Gemini AI.

