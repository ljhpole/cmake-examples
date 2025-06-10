"""
Data models for the Video RAG system
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status of video processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoMetadata(BaseModel):
    """Metadata for a processed video"""
    
    id: str = Field(..., description="Unique identifier for the video")
    filename: str = Field(..., description="Original filename")
    title: Optional[str] = Field(None, description="Video title")
    description: Optional[str] = Field(None, description="Video description")
    duration: float = Field(..., description="Duration in seconds")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Video format")
    resolution: str = Field(..., description="Video resolution (e.g., 1920x1080)")
    fps: float = Field(..., description="Frames per second")
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    tags: List[str] = Field(default_factory=list, description="Content tags")
    
    # CMake-specific metadata
    cmake_topics: List[str] = Field(default_factory=list, description="CMake topics covered")
    difficulty_level: Optional[str] = Field(None, description="Beginner, Intermediate, Advanced")
    code_examples: List[str] = Field(default_factory=list, description="Code snippets found")


class ProcessingResult(BaseModel):
    """Result of video processing"""
    
    video_id: str
    status: ProcessingStatus
    transcript: Optional[str] = None
    summary: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    timestamps: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_code: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Gemini analysis results
    gemini_analysis: Optional[Dict[str, Any]] = None
    visual_elements: List[str] = Field(default_factory=list)
    technical_concepts: List[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Result of a query against the video database"""
    
    query: str
    response: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float = Field(..., description="Query processing time in seconds")
    
    # Source video information
    relevant_videos: List[str] = Field(default_factory=list, description="Video IDs")
    timestamps: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChunkMetadata(BaseModel):
    """Metadata for text chunks stored in vector database"""
    
    video_id: str
    chunk_id: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    content_type: str = Field(..., description="transcript, summary, visual_description, etc.")
    chunk_text: str
    embedding_model: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UploadRequest(BaseModel):
    """Request model for video upload"""
    
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    cmake_topics: List[str] = Field(default_factory=list)
    difficulty_level: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for querying the video database"""
    
    query: str = Field(..., min_length=1, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=20)
    include_timestamps: bool = Field(default=True)
    filter_by_topics: List[str] = Field(default_factory=list)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

