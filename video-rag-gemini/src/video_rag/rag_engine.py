"""
Core RAG engine for video content retrieval and generation
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import uuid

from .config import get_config
from .core.models import QueryResult, VideoMetadata, ProcessingResult
from .core.exceptions import QueryError
from .vector_store import VectorStore
from .gemini_client import GeminiClient
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class VideoRAGEngine:
    """Main RAG engine for video content processing and querying"""
    
    def __init__(self):
        self.config = get_config()
        self.vector_store = VectorStore()
        self.gemini_client = GeminiClient()
        self.video_processor = VideoProcessor()
        
        # Cache for frequently accessed data
        self._video_cache = {}
        self._query_cache = {}
        
        logger.info("VideoRAGEngine initialized")
    
    async def process_video(self, video_path: str, metadata: Dict = None) -> str:
        """
        Process a video and store it in the vector database
        
        Args:
            video_path: Path to the video file
            metadata: Optional metadata dictionary
            
        Returns:
            Video ID for the processed video
        """
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Process the video
            processing_result = await self.video_processor.process_video(video_path, metadata)
            
            if processing_result.status.value == "failed":
                raise QueryError(f"Video processing failed: {processing_result.error_message}")
            
            # Create video metadata
            video_metadata = VideoMetadata(
                id=processing_result.video_id,
                filename=video_path.split('/')[-1],
                title=metadata.get('title') if metadata else None,
                description=metadata.get('description') if metadata else None,
                duration=0,  # Will be updated by video processor
                file_size=0,  # Will be updated by video processor
                format="unknown",  # Will be updated by video processor
                resolution="unknown",  # Will be updated by video processor
                fps=0,  # Will be updated by video processor
                tags=metadata.get('tags', []) if metadata else [],
                cmake_topics=metadata.get('cmake_topics', []) if metadata else [],
                difficulty_level=metadata.get('difficulty_level') if metadata else None
            )
            
            # Store in vector database
            await self.vector_store.store_video_content(video_metadata, processing_result)
            
            # Cache the video metadata
            self._video_cache[processing_result.video_id] = {
                'metadata': video_metadata,
                'processing_result': processing_result
            }
            
            logger.info(f"Video processed successfully: {processing_result.video_id}")
            return processing_result.video_id
            
        except Exception as e:
            logger.error(f"Failed to process video: {str(e)}")
            raise QueryError(f"Failed to process video: {str(e)}")
    
    async def query(self, query_text: str, max_results: int = 5, 
                   filter_by_topics: List[str] = None, 
                   min_confidence: float = 0.0) -> QueryResult:
        """
        Query the video database and generate a response
        
        Args:
            query_text: User's query
            max_results: Maximum number of results to retrieve
            filter_by_topics: Optional topic filters
            min_confidence: Minimum confidence threshold
            
        Returns:
            QueryResult with response and sources
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query_text}")
            
            # Check cache first
            cache_key = self._get_cache_key(query_text, max_results, filter_by_topics, min_confidence)
            if cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                logger.info("Returning cached query result")
                return cached_result
            
            # Retrieve relevant content
            relevant_content = await self.vector_store.search_similar_content(
                query=query_text,
                max_results=max_results * 2,  # Get more results for better filtering
                filter_by_topics=filter_by_topics
            )
            
            # Filter by confidence
            filtered_content = [
                content for content in relevant_content 
                if content['similarity_score'] >= min_confidence
            ]
            
            if not filtered_content:
                return QueryResult(
                    query=query_text,
                    response="I couldn't find any relevant information for your query. Please try rephrasing your question or check if there are videos covering this topic.",
                    confidence_score=0.0,
                    processing_time=time.time() - start_time
                )
            
            # Prepare context for generation
            context_chunks = []
            sources = []
            relevant_videos = set()
            
            for content in filtered_content[:max_results]:
                context_chunks.append(content['content'])
                
                # Build source information
                source_info = {
                    'video_id': content['metadata'].get('video_id', 'unknown'),
                    'content_type': content['metadata'].get('content_type', 'unknown'),
                    'similarity_score': content['similarity_score'],
                    'content_preview': content['content'][:200] + "..." if len(content['content']) > 200 else content['content']
                }
                sources.append(source_info)
                relevant_videos.add(content['metadata'].get('video_id', 'unknown'))
            
            # Generate response using Gemini
            response = await self.gemini_client.answer_query(query_text, context_chunks)
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(filtered_content, query_text)
            
            # Create result
            result = QueryResult(
                query=query_text,
                response=response,
                confidence_score=confidence_score,
                sources=sources,
                processing_time=time.time() - start_time,
                relevant_videos=list(relevant_videos)
            )
            
            # Cache the result
            self._query_cache[cache_key] = result
            
            # Limit cache size
            if len(self._query_cache) > 100:
                # Remove oldest entries
                oldest_keys = list(self._query_cache.keys())[:20]
                for key in oldest_keys:
                    del self._query_cache[key]
            
            logger.info(f"Query processed successfully in {result.processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return QueryResult(
                query=query_text,
                response=f"An error occurred while processing your query: {str(e)}",
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    async def get_video_summary(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information for a specific video
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video summary information
        """
        try:
            # Check cache first
            if video_id in self._video_cache:
                cached_data = self._video_cache[video_id]
                return {
                    'video_id': video_id,
                    'metadata': cached_data['metadata'].dict(),
                    'summary': cached_data['processing_result'].summary,
                    'key_points': cached_data['processing_result'].key_points,
                    'technical_concepts': cached_data['processing_result'].technical_concepts
                }
            
            # Retrieve from vector store
            video_metadata = await self.vector_store.get_video_metadata(video_id)
            if not video_metadata:
                return None
            
            return {
                'video_id': video_id,
                'metadata': video_metadata['metadata'],
                'content': video_metadata['content']
            }
            
        except Exception as e:
            logger.error(f"Failed to get video summary: {str(e)}")
            return None
    
    async def list_videos(self, topic_filter: List[str] = None) -> List[Dict[str, Any]]:
        """
        List all videos in the database
        
        Args:
            topic_filter: Optional topic filter
            
        Returns:
            List of video information
        """
        try:
            # This is a simplified implementation
            # In a real system, you'd want proper pagination and filtering
            videos = []
            
            for video_id, cached_data in self._video_cache.items():
                metadata = cached_data['metadata']
                
                # Apply topic filter if specified
                if topic_filter:
                    if not any(topic in metadata.cmake_topics for topic in topic_filter):
                        continue
                
                video_info = {
                    'video_id': video_id,
                    'title': metadata.title or metadata.filename,
                    'description': metadata.description,
                    'duration': metadata.duration,
                    'cmake_topics': metadata.cmake_topics,
                    'difficulty_level': metadata.difficulty_level,
                    'tags': metadata.tags
                }
                videos.append(video_info)
            
            return videos
            
        except Exception as e:
            logger.error(f"Failed to list videos: {str(e)}")
            return []
    
    async def delete_video(self, video_id: str) -> bool:
        """
        Delete a video and all its associated data
        
        Args:
            video_id: Video identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from vector store
            await self.vector_store.delete_video_content(video_id)
            
            # Remove from cache
            if video_id in self._video_cache:
                del self._video_cache[video_id]
            
            # Clear related query cache entries
            keys_to_remove = []
            for key in self._query_cache:
                if video_id in str(self._query_cache[key].relevant_videos):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._query_cache[key]
            
            logger.info(f"Video {video_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete video: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                'total_videos': len(self._video_cache),
                'cached_queries': len(self._query_cache),
                'vector_database': vector_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {'error': str(e)}
    
    def _get_cache_key(self, query: str, max_results: int, 
                      filter_by_topics: List[str], min_confidence: float) -> str:
        """Generate cache key for query"""
        topics_str = ",".join(sorted(filter_by_topics)) if filter_by_topics else ""
        return f"{query}|{max_results}|{topics_str}|{min_confidence}"
    
    def _calculate_confidence_score(self, content_results: List[Dict], query: str) -> float:
        """Calculate overall confidence score for the query results"""
        if not content_results:
            return 0.0
        
        # Use the average similarity score of top results
        top_scores = [result['similarity_score'] for result in content_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Apply some heuristics
        # Boost confidence if we have multiple relevant results
        if len(content_results) >= 3:
            avg_score *= 1.1
        
        # Boost confidence if results are from multiple videos
        unique_videos = set(result['metadata'].get('video_id') for result in content_results)
        if len(unique_videos) >= 2:
            avg_score *= 1.05
        
        # Ensure score is between 0 and 1
        return min(1.0, max(0.0, avg_score))
    
    async def suggest_related_queries(self, query: str) -> List[str]:
        """
        Suggest related queries based on the current query
        
        Args:
            query: Current query
            
        Returns:
            List of suggested related queries
        """
        try:
            # Get relevant content for the query
            relevant_content = await self.vector_store.search_similar_content(
                query=query,
                max_results=10
            )
            
            # Extract topics and concepts from the results
            topics = set()
            concepts = set()
            
            for content in relevant_content:
                metadata = content['metadata']
                if 'cmake_topics' in metadata:
                    topics.update(metadata.get('cmake_topics', []))
                if 'content_type' in metadata:
                    content_type = metadata['content_type']
                    if content_type == 'key_point':
                        # Extract concepts from key points
                        text = content['content'].lower()
                        cmake_terms = ['build', 'compile', 'link', 'target', 'library', 'executable']
                        for term in cmake_terms:
                            if term in text:
                                concepts.add(term)
            
            # Generate suggested queries
            suggestions = []
            
            # Topic-based suggestions
            for topic in list(topics)[:3]:
                suggestions.append(f"How to use {topic} in CMake?")
                suggestions.append(f"Best practices for {topic}")
            
            # Concept-based suggestions
            for concept in list(concepts)[:2]:
                suggestions.append(f"Examples of {concept} in CMake")
            
            # Generic helpful suggestions
            suggestions.extend([
                "How to debug CMake build issues?",
                "CMake best practices for beginners",
                "How to structure a CMake project?"
            ])
            
            # Remove duplicates and limit
            unique_suggestions = list(set(suggestions))
            return unique_suggestions[:5]
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {str(e)}")
            return [
                "How to get started with CMake?",
                "CMake build system basics",
                "How to create a CMake project?"
            ]

