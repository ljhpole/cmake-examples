"""
API routes for Video RAG system
"""

import os
import tempfile
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse

from ..rag_engine import VideoRAGEngine
from ..core.models import QueryRequest, QueryResult, UploadRequest
from ..core.exceptions import VideoRAGException
from ..config import get_config

logger = logging.getLogger(__name__)
router = APIRouter()

# Global RAG engine instance (will be injected)
def get_rag_engine() -> VideoRAGEngine:
    """Dependency injection for RAG engine"""
    # This will be overridden by the main app
    pass


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    cmake_topics: Optional[str] = Form(None),
    difficulty_level: Optional[str] = Form(None),
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Upload and process a video file
    
    Args:
        file: Video file to upload
        title: Optional video title
        description: Optional video description
        cmake_topics: Comma-separated list of CMake topics
        difficulty_level: Difficulty level (Beginner, Intermediate, Advanced)
        
    Returns:
        Processing result with video ID
    """
    try:
        logger.info(f"Received video upload: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        config = get_config()
        max_size = config.max_video_size_mb * 1024 * 1024
        
        # Read file content
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {config.max_video_size_mb}MB"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Prepare metadata
            metadata = {}
            if title:
                metadata['title'] = title
            if description:
                metadata['description'] = description
            if cmake_topics:
                metadata['cmake_topics'] = [topic.strip() for topic in cmake_topics.split(',') if topic.strip()]
            if difficulty_level:
                metadata['difficulty_level'] = difficulty_level
            
            # Process video
            video_id = await rag_engine.process_video(temp_path, metadata)
            
            return {
                "success": True,
                "video_id": video_id,
                "message": "Video processed successfully",
                "filename": file.filename
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except VideoRAGException as e:
        logger.error(f"Video processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during video upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/query", response_model=QueryResult)
async def query_videos(
    request: QueryRequest,
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Query the video database
    
    Args:
        request: Query request with parameters
        
    Returns:
        Query result with response and sources
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        result = await rag_engine.query(
            query_text=request.query,
            max_results=request.max_results,
            filter_by_topics=request.filter_by_topics if request.filter_by_topics else None,
            min_confidence=request.min_confidence
        )
        
        return result
        
    except VideoRAGException as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/videos")
async def list_videos(
    topic_filter: Optional[str] = None,
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    List all videos in the database
    
    Args:
        topic_filter: Optional comma-separated list of topics to filter by
        
    Returns:
        List of videos with metadata
    """
    try:
        topics = None
        if topic_filter:
            topics = [topic.strip() for topic in topic_filter.split(',') if topic.strip()]
        
        videos = await rag_engine.list_videos(topic_filter=topics)
        
        return {
            "success": True,
            "videos": videos,
            "count": len(videos)
        }
        
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/videos/{video_id}")
async def get_video_summary(
    video_id: str,
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Get summary information for a specific video
    
    Args:
        video_id: Video identifier
        
    Returns:
        Video summary and metadata
    """
    try:
        summary = await rag_engine.get_video_summary(video_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {
            "success": True,
            "video_summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: str,
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Delete a video and all its associated data
    
    Args:
        video_id: Video identifier
        
    Returns:
        Deletion result
    """
    try:
        success = await rag_engine.delete_video(video_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Video not found or deletion failed")
        
        return {
            "success": True,
            "message": f"Video {video_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/stats")
async def get_system_stats(
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Get system statistics
    
    Returns:
        System statistics including video count, cache info, etc.
    """
    try:
        stats = rag_engine.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/suggestions")
async def get_query_suggestions(
    query: str,
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Get suggested related queries
    
    Args:
        query: Current query to base suggestions on
        
    Returns:
        List of suggested queries
    """
    try:
        suggestions = await rag_engine.suggest_related_queries(query)
        
        return {
            "success": True,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/topics")
async def get_available_topics(
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Get list of available CMake topics from processed videos
    
    Returns:
        List of unique topics
    """
    try:
        videos = await rag_engine.list_videos()
        
        # Extract unique topics
        topics = set()
        for video in videos:
            topics.update(video.get('cmake_topics', []))
        
        return {
            "success": True,
            "topics": sorted(list(topics)),
            "count": len(topics)
        }
        
    except Exception as e:
        logger.error(f"Error getting topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/batch-query")
async def batch_query(
    queries: List[str],
    max_results: int = 5,
    rag_engine: VideoRAGEngine = Depends(get_rag_engine)
):
    """
    Process multiple queries in batch
    
    Args:
        queries: List of queries to process
        max_results: Maximum results per query
        
    Returns:
        List of query results
    """
    try:
        if len(queries) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 queries per batch")
        
        results = []
        for query in queries:
            result = await rag_engine.query(
                query_text=query,
                max_results=max_results
            )
            results.append(result)
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

