"""
FastAPI application for Video RAG system
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..config import get_config, validate_config
from ..rag_engine import VideoRAGEngine
from ..core.models import QueryRequest, QueryResult, UploadRequest
from ..core.exceptions import VideoRAGException
from .routes import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG engine instance
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global rag_engine
    
    # Startup
    logger.info("Starting Video RAG API...")
    
    # Validate configuration
    if not validate_config():
        raise RuntimeError("Configuration validation failed")
    
    # Initialize RAG engine
    try:
        rag_engine = VideoRAGEngine()
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {str(e)}")
        raise RuntimeError(f"Failed to initialize RAG engine: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Video RAG API...")


# Create FastAPI app
app = FastAPI(
    title="Video RAG System",
    description="Retrieval-Augmented Generation system for video content using Gemini API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def get_rag_engine() -> VideoRAGEngine:
    """Dependency to get RAG engine instance"""
    if rag_engine is None:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")
    return rag_engine


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint serving the web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video RAG System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .section {
                margin-bottom: 30px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
            .section h2 {
                color: #555;
                margin-top: 0;
            }
            input, textarea, button, select {
                width: 100%;
                padding: 10px;
                margin: 5px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                cursor: pointer;
                font-weight: bold;
            }
            button:hover {
                background-color: #0056b3;
            }
            .response {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                border-left: 4px solid #007bff;
                margin-top: 10px;
                white-space: pre-wrap;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
                border-left-color: #dc3545;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 4px;
                text-align: center;
            }
            .stat-number {
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 Video RAG System with Gemini AI</h1>
            
            <div class="section">
                <h2>📤 Upload Video</h2>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="videoFile" accept="video/*" required>
                    <input type="text" id="videoTitle" placeholder="Video Title (optional)">
                    <textarea id="videoDescription" placeholder="Video Description (optional)" rows="3"></textarea>
                    <input type="text" id="videoTopics" placeholder="CMake Topics (comma-separated)">
                    <select id="videoDifficulty">
                        <option value="">Select Difficulty Level</option>
                        <option value="Beginner">Beginner</option>
                        <option value="Intermediate">Intermediate</option>
                        <option value="Advanced">Advanced</option>
                    </select>
                    <button type="submit">Upload and Process Video</button>
                </form>
                <div id="uploadResponse" class="response" style="display:none;"></div>
                <div id="uploadLoading" class="loading">Processing video... This may take a few minutes.</div>
            </div>
            
            <div class="section">
                <h2>❓ Ask Questions</h2>
                <form id="queryForm">
                    <textarea id="queryText" placeholder="Ask a question about your videos..." rows="3" required></textarea>
                    <input type="number" id="maxResults" placeholder="Max Results (default: 5)" min="1" max="20" value="5">
                    <input type="text" id="topicFilter" placeholder="Filter by Topics (comma-separated, optional)">
                    <button type="submit">Ask Question</button>
                </form>
                <div id="queryResponse" class="response" style="display:none;"></div>
                <div id="queryLoading" class="loading">Searching and generating response...</div>
            </div>
            
            <div class="section">
                <h2>📊 System Statistics</h2>
                <button onclick="loadStats()">Refresh Stats</button>
                <div id="statsContainer" class="stats"></div>
            </div>
            
            <div class="section">
                <h2>📋 API Documentation</h2>
                <p>Access the interactive API documentation at:</p>
                <ul>
                    <li><a href="/docs" target="_blank">Swagger UI</a></li>
                    <li><a href="/redoc" target="_blank">ReDoc</a></li>
                </ul>
            </div>
        </div>

        <script>
            // Upload form handler
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('videoFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a video file');
                    return;
                }
                
                formData.append('file', file);
                formData.append('title', document.getElementById('videoTitle').value);
                formData.append('description', document.getElementById('videoDescription').value);
                formData.append('cmake_topics', document.getElementById('videoTopics').value);
                formData.append('difficulty_level', document.getElementById('videoDifficulty').value);
                
                const responseDiv = document.getElementById('uploadResponse');
                const loadingDiv = document.getElementById('uploadLoading');
                
                try {
                    loadingDiv.style.display = 'block';
                    responseDiv.style.display = 'none';
                    
                    const response = await fetch('/api/v1/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        responseDiv.innerHTML = `✅ Video uploaded successfully!\\nVideo ID: ${result.video_id}\\nProcessing completed in ${result.processing_time?.toFixed(2)} seconds`;
                        responseDiv.className = 'response';
                        document.getElementById('uploadForm').reset();
                    } else {
                        responseDiv.innerHTML = `❌ Upload failed: ${result.detail || 'Unknown error'}`;
                        responseDiv.className = 'response error';
                    }
                } catch (error) {
                    responseDiv.innerHTML = `❌ Upload failed: ${error.message}`;
                    responseDiv.className = 'response error';
                } finally {
                    loadingDiv.style.display = 'none';
                    responseDiv.style.display = 'block';
                }
            });
            
            // Query form handler
            document.getElementById('queryForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const query = document.getElementById('queryText').value;
                const maxResults = parseInt(document.getElementById('maxResults').value) || 5;
                const topicFilter = document.getElementById('topicFilter').value;
                
                const responseDiv = document.getElementById('queryResponse');
                const loadingDiv = document.getElementById('queryLoading');
                
                try {
                    loadingDiv.style.display = 'block';
                    responseDiv.style.display = 'none';
                    
                    const requestBody = {
                        query: query,
                        max_results: maxResults,
                        filter_by_topics: topicFilter ? topicFilter.split(',').map(t => t.trim()) : []
                    };
                    
                    const response = await fetch('/api/v1/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(requestBody)
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let responseText = `🤖 **Answer:**\\n${result.response}\\n\\n`;
                        responseText += `📊 **Confidence:** ${(result.confidence_score * 100).toFixed(1)}%\\n`;
                        responseText += `⏱️ **Processing Time:** ${result.processing_time?.toFixed(2)} seconds\\n\\n`;
                        
                        if (result.sources && result.sources.length > 0) {
                            responseText += `📚 **Sources:**\\n`;
                            result.sources.forEach((source, index) => {
                                responseText += `${index + 1}. ${source.content_type} (${(source.similarity_score * 100).toFixed(1)}% match)\\n`;
                                responseText += `   Preview: ${source.content_preview}\\n\\n`;
                            });
                        }
                        
                        responseDiv.innerHTML = responseText;
                        responseDiv.className = 'response';
                    } else {
                        responseDiv.innerHTML = `❌ Query failed: ${result.detail || 'Unknown error'}`;
                        responseDiv.className = 'response error';
                    }
                } catch (error) {
                    responseDiv.innerHTML = `❌ Query failed: ${error.message}`;
                    responseDiv.className = 'response error';
                } finally {
                    loadingDiv.style.display = 'none';
                    responseDiv.style.display = 'block';
                }
            });
            
            // Load system statistics
            async function loadStats() {
                try {
                    const response = await fetch('/api/v1/stats');
                    const stats = await response.json();
                    
                    const container = document.getElementById('statsContainer');
                    container.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-number">${stats.total_videos || 0}</div>
                            <div>Total Videos</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.cached_queries || 0}</div>
                            <div>Cached Queries</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.vector_database?.content_chunks || 0}</div>
                            <div>Content Chunks</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">${stats.vector_database?.video_metadata_entries || 0}</div>
                            <div>Metadata Entries</div>
                        </div>
                    `;
                } catch (error) {
                    document.getElementById('statsContainer').innerHTML = `<div class="error">Failed to load stats: ${error.message}</div>`;
                }
            }
            
            // Load stats on page load
            loadStats();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "video-rag-api"}


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug
    )

