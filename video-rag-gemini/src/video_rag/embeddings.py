"""
Embedding generation for video content using sentence transformers
"""

import logging
import asyncio
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .config import get_config
from .core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text content using sentence transformers"""
    
    def __init__(self):
        self.config = get_config()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            
            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            
            logger.info(f"Embedding model loaded on device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise VectorStoreError(f"Failed to load embedding model: {str(e)}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding
        """
        if not self.model:
            raise VectorStoreError("Embedding model not loaded")
        
        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self._generate_embedding_sync, 
                text
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise VectorStoreError(f"Failed to generate embedding: {str(e)}")
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embeddings
        """
        if not self.model:
            raise VectorStoreError("Embedding model not loaded")
        
        try:
            # Run batch embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self._generate_embeddings_batch_sync, 
                texts
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise VectorStoreError(f"Failed to generate batch embeddings: {str(e)}")
    
    def _generate_embedding_sync(self, text: str) -> np.ndarray:
        """Synchronous embedding generation for single text"""
        # Preprocess text
        text = self._preprocess_text(text)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        
        return embedding
    
    def _generate_embeddings_batch_sync(self, texts: List[str]) -> np.ndarray:
        """Synchronous embedding generation for batch of texts"""
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            processed_texts, 
            convert_to_tensor=False, 
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation"""
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate if too long (model-dependent, but 512 tokens is common limit)
        max_length = 8000  # Conservative character limit
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if not self.model:
            raise VectorStoreError("Embedding model not loaded")
        
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            return 0.0
    
    async def find_most_similar(self, query_embedding: List[float], 
                               candidate_embeddings: List[List[float]], 
                               top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings to a query embedding
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {str(e)}")
            return []


class CMakeSpecificEmbedding(EmbeddingGenerator):
    """Specialized embedding generator for CMake content"""
    
    def _preprocess_text(self, text: str) -> str:
        """CMake-specific text preprocessing"""
        text = super()._preprocess_text(text)
        
        # Add CMake context for better embeddings
        cmake_keywords = [
            'cmake', 'makefile', 'build', 'compile', 'link', 'target',
            'library', 'executable', 'package', 'configure', 'generate'
        ]
        
        text_lower = text.lower()
        has_cmake_content = any(keyword in text_lower for keyword in cmake_keywords)
        
        if has_cmake_content and not text_lower.startswith('cmake'):
            text = f"CMake build system: {text}"
        
        return text
    
    async def generate_code_embedding(self, code: str, context: str = "") -> List[float]:
        """
        Generate embedding specifically for CMake code
        
        Args:
            code: CMake code snippet
            context: Additional context about the code
            
        Returns:
            Code embedding
        """
        # Format code with context for better embedding
        formatted_text = f"CMake code snippet"
        if context:
            formatted_text += f" ({context})"
        formatted_text += f": {code}"
        
        return await self.generate_embedding(formatted_text)
    
    async def generate_concept_embedding(self, concept: str, definition: str = "") -> List[float]:
        """
        Generate embedding for CMake concepts
        
        Args:
            concept: CMake concept name
            definition: Optional definition or explanation
            
        Returns:
            Concept embedding
        """
        formatted_text = f"CMake concept: {concept}"
        if definition:
            formatted_text += f" - {definition}"
        
        return await self.generate_embedding(formatted_text)

