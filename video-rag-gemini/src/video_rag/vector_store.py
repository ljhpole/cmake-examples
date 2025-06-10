"""
Vector database integration for storing and retrieving video embeddings
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np

from .config import get_config
from .core.models import ChunkMetadata, VideoMetadata, ProcessingResult
from .core.exceptions import VectorStoreError
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self):
        self.config = get_config()
        self.embedding_generator = EmbeddingGenerator()
        self._setup_chroma_client()
        self._setup_collections()
    
    def _setup_chroma_client(self):
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.config.vector_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized at {self.config.vector_db_path}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB client: {str(e)}")
    
    def _setup_collections(self):
        """Setup ChromaDB collections"""
        try:
            # Collection for video content chunks
            self.content_collection = self.client.get_or_create_collection(
                name="video_content",
                metadata={"description": "Video content chunks with embeddings"}
            )
            
            # Collection for video metadata
            self.metadata_collection = self.client.get_or_create_collection(
                name="video_metadata",
                metadata={"description": "Video metadata and summaries"}
            )
            
            logger.info("ChromaDB collections initialized")
        except Exception as e:
            raise VectorStoreError(f"Failed to setup collections: {str(e)}")
    
    async def store_video_content(self, video_metadata: VideoMetadata, processing_result: ProcessingResult):
        """
        Store video content and metadata in vector database
        
        Args:
            video_metadata: Video metadata
            processing_result: Processing results with extracted content
        """
        try:
            logger.info(f"Storing video content for video {video_metadata.id}")
            
            # Store video metadata
            await self._store_video_metadata(video_metadata, processing_result)
            
            # Chunk and store transcript
            if processing_result.transcript:
                await self._store_transcript_chunks(video_metadata.id, processing_result.transcript)
            
            # Store summary
            if processing_result.summary:
                await self._store_summary(video_metadata.id, processing_result.summary)
            
            # Store key points
            if processing_result.key_points:
                await self._store_key_points(video_metadata.id, processing_result.key_points)
            
            # Store code examples
            if processing_result.extracted_code:
                await self._store_code_examples(video_metadata.id, processing_result.extracted_code)
            
            logger.info(f"Successfully stored content for video {video_metadata.id}")
            
        except Exception as e:
            logger.error(f"Failed to store video content: {str(e)}")
            raise VectorStoreError(f"Failed to store video content: {str(e)}")
    
    async def _store_video_metadata(self, video_metadata: VideoMetadata, processing_result: ProcessingResult):
        """Store video metadata in the metadata collection"""
        try:
            # Create metadata document
            metadata_text = f"""
            Title: {video_metadata.title or video_metadata.filename}
            Description: {video_metadata.description or 'No description'}
            Duration: {video_metadata.duration} seconds
            Topics: {', '.join(video_metadata.cmake_topics)}
            Difficulty: {video_metadata.difficulty_level or 'Unknown'}
            Tags: {', '.join(video_metadata.tags)}
            Summary: {processing_result.summary or 'No summary available'}
            """
            
            # Generate embedding
            embedding = await self.embedding_generator.generate_embedding(metadata_text)
            
            # Store in collection
            self.metadata_collection.add(
                documents=[metadata_text],
                embeddings=[embedding],
                metadatas=[{
                    "video_id": video_metadata.id,
                    "filename": video_metadata.filename,
                    "title": video_metadata.title or "",
                    "duration": video_metadata.duration,
                    "cmake_topics": video_metadata.cmake_topics,
                    "difficulty_level": video_metadata.difficulty_level or "",
                    "content_type": "metadata"
                }],
                ids=[f"metadata_{video_metadata.id}"]
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to store video metadata: {str(e)}")
    
    async def _store_transcript_chunks(self, video_id: str, transcript: str):
        """Chunk and store transcript with embeddings"""
        try:
            # Split transcript into chunks
            chunks = self._chunk_text(transcript)
            
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                # Generate embedding for chunk
                embedding = await self.embedding_generator.generate_embedding(chunk)
                
                documents.append(chunk)
                embeddings.append(embedding)
                metadatas.append({
                    "video_id": video_id,
                    "chunk_index": i,
                    "content_type": "transcript",
                    "chunk_size": len(chunk)
                })
                ids.append(f"transcript_{video_id}_{i}")
            
            # Store chunks in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                
                self.content_collection.add(
                    documents=documents[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end]
                )
            
            logger.info(f"Stored {len(chunks)} transcript chunks for video {video_id}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to store transcript chunks: {str(e)}")
    
    async def _store_summary(self, video_id: str, summary: str):
        """Store video summary with embedding"""
        try:
            embedding = await self.embedding_generator.generate_embedding(summary)
            
            self.content_collection.add(
                documents=[summary],
                embeddings=[embedding],
                metadatas=[{
                    "video_id": video_id,
                    "content_type": "summary"
                }],
                ids=[f"summary_{video_id}"]
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to store summary: {str(e)}")
    
    async def _store_key_points(self, video_id: str, key_points: List[str]):
        """Store key points with embeddings"""
        try:
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, point in enumerate(key_points):
                embedding = await self.embedding_generator.generate_embedding(point)
                
                documents.append(point)
                embeddings.append(embedding)
                metadatas.append({
                    "video_id": video_id,
                    "content_type": "key_point",
                    "point_index": i
                })
                ids.append(f"keypoint_{video_id}_{i}")
            
            self.content_collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to store key points: {str(e)}")
    
    async def _store_code_examples(self, video_id: str, code_examples: List[str]):
        """Store code examples with embeddings"""
        try:
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, code in enumerate(code_examples):
                # Add context to code for better embedding
                code_with_context = f"CMake code example: {code}"
                embedding = await self.embedding_generator.generate_embedding(code_with_context)
                
                documents.append(code)
                embeddings.append(embedding)
                metadatas.append({
                    "video_id": video_id,
                    "content_type": "code_example",
                    "code_index": i
                })
                ids.append(f"code_{video_id}_{i}")
            
            self.content_collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to store code examples: {str(e)}")
    
    async def search_similar_content(self, query: str, max_results: int = 5, 
                                   filter_by_topics: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content using vector similarity
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            filter_by_topics: Optional list of topics to filter by
            
        Returns:
            List of similar content with metadata
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Build where clause for filtering
            where_clause = None
            if filter_by_topics:
                where_clause = {"cmake_topics": {"$in": filter_by_topics}}
            
            # Search in content collection
            content_results = self.content_collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Search in metadata collection
            metadata_results = self.metadata_collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results // 2,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Combine and format results
            all_results = []
            
            # Process content results
            if content_results['documents']:
                for i, doc in enumerate(content_results['documents'][0]):
                    result = {
                        "content": doc,
                        "metadata": content_results['metadatas'][0][i],
                        "similarity_score": 1 - content_results['distances'][0][i],  # Convert distance to similarity
                        "source": "content"
                    }
                    all_results.append(result)
            
            # Process metadata results
            if metadata_results['documents']:
                for i, doc in enumerate(metadata_results['documents'][0]):
                    result = {
                        "content": doc,
                        "metadata": metadata_results['metadatas'][0][i],
                        "similarity_score": 1 - metadata_results['distances'][0][i],
                        "source": "metadata"
                    }
                    all_results.append(result)
            
            # Sort by similarity score and return top results
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return all_results[:max_results]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorStoreError(f"Search failed: {str(e)}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding"""
        max_chunk_size = self.config.max_chunk_size
        overlap = self.config.chunk_overlap
        
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, start + max_chunk_size // 2, -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific video"""
        try:
            results = self.metadata_collection.get(
                ids=[f"metadata_{video_id}"],
                include=["documents", "metadatas"]
            )
            
            if results['documents']:
                return {
                    "content": results['documents'][0],
                    "metadata": results['metadatas'][0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get video metadata: {str(e)}")
            return None
    
    async def delete_video_content(self, video_id: str):
        """Delete all content for a specific video"""
        try:
            # Get all content for this video
            content_results = self.content_collection.get(
                where={"video_id": video_id},
                include=["metadatas"]
            )
            
            metadata_results = self.metadata_collection.get(
                where={"video_id": video_id},
                include=["metadatas"]
            )
            
            # Delete from content collection
            if content_results['ids']:
                self.content_collection.delete(ids=content_results['ids'])
            
            # Delete from metadata collection
            if metadata_results['ids']:
                self.metadata_collection.delete(ids=metadata_results['ids'])
            
            logger.info(f"Deleted all content for video {video_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete video content: {str(e)}")
            raise VectorStoreError(f"Failed to delete video content: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            content_count = self.content_collection.count()
            metadata_count = self.metadata_collection.count()
            
            return {
                "content_chunks": content_count,
                "video_metadata_entries": metadata_count,
                "total_entries": content_count + metadata_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}

