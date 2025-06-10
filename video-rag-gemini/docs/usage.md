# Video RAG System Usage Guide

This guide provides detailed instructions on how to use the Video RAG system for processing and querying CMake tutorial videos.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Video Processing](#video-processing)
3. [Querying Videos](#querying-videos)
4. [API Reference](#api-reference)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using the Video RAG system, ensure you have:

1. **Python 3.8+** installed
2. **CMake 3.16+** (for building)
3. **FFmpeg** (for video processing)
4. **Google Gemini API key**

### Initial Setup

1. **Configure your environment**:
   ```bash
   cp .env.template .env
   # Edit .env and add your GEMINI_API_KEY
   ```

2. **Install dependencies**:
   ```bash
   make install_python_deps
   # or manually: pip install -r requirements.txt
   ```

3. **Setup data directories**:
   ```bash
   make setup_environment
   ```

4. **Start the server**:
   ```bash
   make run_server
   ```

## Video Processing

### Supported Video Formats

The system supports the following video formats:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)

### Upload via Web Interface

1. Open http://localhost:8000 in your browser
2. Navigate to the "Upload Video" section
3. Select your video file
4. Fill in optional metadata:
   - **Title**: Descriptive title for the video
   - **Description**: Brief description of the content
   - **CMake Topics**: Comma-separated topics (e.g., "build system, targets, libraries")
   - **Difficulty Level**: Beginner, Intermediate, or Advanced

5. Click "Upload and Process Video"
6. Wait for processing to complete (may take several minutes)

### Upload via API

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@cmake_tutorial.mp4" \
  -F "title=CMake Build System Basics" \
  -F "description=Introduction to CMake fundamentals" \
  -F "cmake_topics=build system,targets,executables" \
  -F "difficulty_level=Beginner"
```

### Upload via Python

```python
import asyncio
from video_rag import VideoRAGEngine

async def upload_video():
    rag = VideoRAGEngine()
    
    video_id = await rag.process_video(
        "path/to/video.mp4",
        metadata={
            "title": "CMake Tutorial",
            "description": "Learn CMake basics",
            "cmake_topics": ["basics", "targets"],
            "difficulty_level": "Beginner"
        }
    )
    
    print(f"Video processed: {video_id}")

asyncio.run(upload_video())
```

### Processing Pipeline

When you upload a video, the system performs the following steps:

1. **Video Validation**: Checks format, size, and accessibility
2. **Metadata Extraction**: Extracts duration, resolution, frame rate
3. **Audio Transcription**: Uses Whisper to generate transcript
4. **Visual Analysis**: Extracts key frames for analysis
5. **Gemini Analysis**: Analyzes video content using Gemini API
6. **Content Chunking**: Splits content into searchable chunks
7. **Embedding Generation**: Creates vector embeddings for search
8. **Storage**: Stores all data in the vector database

## Querying Videos

### Query via Web Interface

1. Navigate to the "Ask Questions" section
2. Enter your question in natural language
3. Optionally set:
   - **Max Results**: Number of results to return (1-20)
   - **Topic Filter**: Comma-separated topics to filter by
4. Click "Ask Question"
5. Review the response, confidence score, and sources

### Query via API

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a static library in CMake?",
    "max_results": 5,
    "filter_by_topics": ["library", "static"],
    "min_confidence": 0.1
  }'
```

### Query via Python

```python
import asyncio
from video_rag import VideoRAGEngine

async def query_videos():
    rag = VideoRAGEngine()
    
    result = await rag.query(
        query_text="How do I link libraries in CMake?",
        max_results=5,
        filter_by_topics=["library", "linking"],
        min_confidence=0.2
    )
    
    print(f"Answer: {result.response}")
    print(f"Confidence: {result.confidence_score:.2%}")
    print(f"Sources: {len(result.sources)}")

asyncio.run(query_videos())
```

### Query Types

The system handles various types of queries:

#### Basic Questions
- "What is CMake?"
- "How do I create a CMake project?"
- "What is a CMakeLists.txt file?"

#### How-to Questions
- "How do I create an executable?"
- "How do I link libraries?"
- "How do I set include directories?"

#### Comparison Questions
- "What's the difference between add_executable and add_library?"
- "When should I use PUBLIC vs PRIVATE?"

#### Troubleshooting Questions
- "Why is my CMake build failing?"
- "How do I fix linking errors?"
- "What does 'target not found' mean?"

#### Best Practices
- "What are CMake best practices?"
- "How should I structure my CMake project?"
- "What are common CMake mistakes?"

## API Reference

### Core Endpoints

#### POST /api/v1/upload
Upload and process a video file.

**Parameters:**
- `file`: Video file (multipart/form-data)
- `title`: Optional title
- `description`: Optional description
- `cmake_topics`: Comma-separated topics
- `difficulty_level`: Beginner/Intermediate/Advanced

**Response:**
```json
{
  "success": true,
  "video_id": "uuid",
  "message": "Video processed successfully",
  "filename": "video.mp4"
}
```

#### POST /api/v1/query
Query the video database.

**Request Body:**
```json
{
  "query": "How do I create a library?",
  "max_results": 5,
  "filter_by_topics": ["library"],
  "min_confidence": 0.1
}
```

**Response:**
```json
{
  "query": "How do I create a library?",
  "response": "To create a library in CMake...",
  "confidence_score": 0.85,
  "sources": [...],
  "processing_time": 1.23,
  "relevant_videos": ["uuid1", "uuid2"]
}
```

#### GET /api/v1/videos
List all processed videos.

**Parameters:**
- `topic_filter`: Optional comma-separated topics

**Response:**
```json
{
  "success": true,
  "videos": [...],
  "count": 5
}
```

#### GET /api/v1/videos/{video_id}
Get detailed information about a specific video.

**Response:**
```json
{
  "success": true,
  "video_summary": {
    "video_id": "uuid",
    "metadata": {...},
    "summary": "Video summary...",
    "key_points": [...]
  }
}
```

#### DELETE /api/v1/videos/{video_id}
Delete a video and all associated data.

**Response:**
```json
{
  "success": true,
  "message": "Video uuid deleted successfully"
}
```

#### GET /api/v1/stats
Get system statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_videos": 10,
    "cached_queries": 25,
    "vector_database": {
      "content_chunks": 150,
      "video_metadata_entries": 10
    }
  }
}
```

### Utility Endpoints

#### GET /api/v1/suggestions
Get query suggestions based on a base query.

**Parameters:**
- `query`: Base query for suggestions

#### GET /api/v1/topics
Get all available CMake topics from processed videos.

#### POST /api/v1/batch-query
Process multiple queries in batch (max 10).

## Advanced Usage

### Custom Embedding Models

You can use different embedding models by modifying the configuration:

```bash
# In .env file
EMBEDDING_MODEL=all-mpnet-base-v2
# or
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2
```

### Batch Processing

Process multiple videos programmatically:

```python
import asyncio
import os
from video_rag import VideoRAGEngine

async def batch_process_videos(video_directory):
    rag = VideoRAGEngine()
    
    for filename in os.listdir(video_directory):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_directory, filename)
            
            try:
                video_id = await rag.process_video(
                    video_path,
                    metadata={
                        "title": filename.replace('.mp4', ''),
                        "cmake_topics": ["general"]
                    }
                )
                print(f"Processed: {filename} -> {video_id}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# Usage
asyncio.run(batch_process_videos("./videos"))
```

### Custom Query Processing

Implement custom query logic:

```python
from video_rag import VideoRAGEngine

class CustomRAGEngine(VideoRAGEngine):
    async def custom_query(self, query: str, context_type: str = "all"):
        # Custom preprocessing
        if "error" in query.lower():
            # Handle error-related queries differently
            filter_topics = ["troubleshooting", "errors"]
        else:
            filter_topics = None
        
        # Use parent query method with custom filters
        result = await self.query(
            query_text=query,
            filter_by_topics=filter_topics,
            max_results=10
        )
        
        # Custom post-processing
        if result.confidence_score < 0.3:
            result.response = f"Low confidence answer: {result.response}"
        
        return result
```

### Integration with External Systems

#### Webhook Integration

Set up webhooks to notify external systems when videos are processed:

```python
import requests
from video_rag import VideoRAGEngine

class WebhookRAGEngine(VideoRAGEngine):
    async def process_video(self, video_path: str, metadata: dict = None):
        # Process video normally
        video_id = await super().process_video(video_path, metadata)
        
        # Send webhook notification
        webhook_url = "https://your-system.com/webhook"
        payload = {
            "event": "video_processed",
            "video_id": video_id,
            "metadata": metadata
        }
        
        try:
            requests.post(webhook_url, json=payload)
        except Exception as e:
            print(f"Webhook failed: {e}")
        
        return video_id
```

## Troubleshooting

### Common Issues

#### 1. Video Processing Fails

**Symptoms:**
- Upload returns error
- Processing status shows "failed"

**Solutions:**
- Check video format is supported
- Verify file size is under limit (default 100MB)
- Ensure FFmpeg is installed
- Check disk space for temporary files

#### 2. Poor Query Results

**Symptoms:**
- Low confidence scores
- Irrelevant responses
- No sources found

**Solutions:**
- Process more videos to build knowledge base
- Use more specific queries
- Check if videos cover the queried topics
- Adjust `min_confidence` parameter

#### 3. Gemini API Errors

**Symptoms:**
- API key errors
- Rate limit exceeded
- Network timeouts

**Solutions:**
- Verify API key is correct and active
- Check API quotas and billing
- Reduce `MAX_REQUESTS_PER_MINUTE` in config
- Ensure stable internet connection

#### 4. Memory Issues

**Symptoms:**
- Out of memory errors
- Slow processing
- System freezes

**Solutions:**
- Reduce video file sizes
- Lower `MAX_CHUNK_SIZE` in config
- Process videos one at a time
- Increase system RAM

### Debug Mode

Enable debug mode for detailed logging:

```bash
# In .env file
DEBUG=true
```

### Log Analysis

Check logs for detailed error information:

```bash
# View recent logs
tail -f logs/video_rag.log

# Search for errors
grep "ERROR" logs/video_rag.log
```

### Performance Optimization

#### For Large Video Collections

1. **Use batch processing** during off-peak hours
2. **Implement caching** for frequently accessed content
3. **Consider distributed processing** for very large collections
4. **Monitor resource usage** and scale accordingly

#### For High Query Volume

1. **Enable query caching** (enabled by default)
2. **Use topic filters** to narrow search scope
3. **Implement rate limiting** for API endpoints
4. **Consider read replicas** for the vector database

### Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Review the API documentation at `/docs`
3. Enable debug logging for detailed error information
4. Provide minimal reproduction steps when reporting issues

---

For more information, see the main [README.md](../README.md) file.

