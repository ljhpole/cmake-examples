"""
Basic tests for the Video RAG system
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

# Import the modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_rag.config import Config, validate_config
from video_rag.core.models import VideoMetadata, ProcessingResult, QueryResult, ProcessingStatus
from video_rag.core.exceptions import VideoRAGException, ProcessingError, QueryError
from video_rag.embeddings import EmbeddingGenerator
from video_rag.gemini_client import GeminiClient, RateLimiter


class TestConfig:
    """Test configuration management"""
    
    def test_config_creation(self):
        """Test basic config creation"""
        # Mock environment variables
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            config = Config()
            assert config.gemini_api_key == 'test_key'
            assert config.max_video_size_mb == 100  # default value
            assert config.api_port == 8000  # default value
    
    def test_config_validation_success(self):
        """Test successful config validation"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('os.makedirs'):
                result = validate_config()
                assert result is True
    
    def test_config_validation_failure(self):
        """Test config validation failure"""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_config()
            assert result is False


class TestModels:
    """Test data models"""
    
    def test_video_metadata_creation(self):
        """Test VideoMetadata model creation"""
        metadata = VideoMetadata(
            id="test_id",
            filename="test.mp4",
            duration=120.5,
            file_size=1024000,
            format="mp4",
            resolution="1920x1080",
            fps=30.0
        )
        
        assert metadata.id == "test_id"
        assert metadata.filename == "test.mp4"
        assert metadata.duration == 120.5
        assert metadata.processing_status == ProcessingStatus.PENDING
    
    def test_processing_result_creation(self):
        """Test ProcessingResult model creation"""
        result = ProcessingResult(
            video_id="test_id",
            status=ProcessingStatus.COMPLETED,
            transcript="Test transcript",
            processing_time=45.2
        )
        
        assert result.video_id == "test_id"
        assert result.status == ProcessingStatus.COMPLETED
        assert result.transcript == "Test transcript"
        assert result.processing_time == 45.2
    
    def test_query_result_creation(self):
        """Test QueryResult model creation"""
        result = QueryResult(
            query="Test query",
            response="Test response",
            confidence_score=0.85,
            processing_time=2.1
        )
        
        assert result.query == "Test query"
        assert result.response == "Test response"
        assert result.confidence_score == 0.85
        assert 0.0 <= result.confidence_score <= 1.0


class TestExceptions:
    """Test custom exceptions"""
    
    def test_video_rag_exception(self):
        """Test base VideoRAGException"""
        exc = VideoRAGException("Test message", "TEST_CODE")
        assert str(exc) == "Test message"
        assert exc.error_code == "TEST_CODE"
    
    def test_processing_error(self):
        """Test ProcessingError exception"""
        exc = ProcessingError("Processing failed")
        assert str(exc) == "Processing failed"
        assert isinstance(exc, VideoRAGException)
    
    def test_query_error(self):
        """Test QueryError exception"""
        exc = QueryError("Query failed")
        assert str(exc) == "Query failed"
        assert isinstance(exc, VideoRAGException)


class TestEmbeddingGenerator:
    """Test embedding generation"""
    
    @pytest.fixture
    def embedding_generator(self):
        """Create a mock embedding generator"""
        with patch('video_rag.embeddings.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_model.get_sentence_embedding_dimension.return_value = 5
            mock_transformer.return_value = mock_model
            
            generator = EmbeddingGenerator()
            return generator
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, embedding_generator):
        """Test single embedding generation"""
        text = "Test text for embedding"
        embedding = await embedding_generator.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 5
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, embedding_generator):
        """Test batch embedding generation"""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await embedding_generator.generate_embeddings_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 5 for emb in embeddings)
    
    def test_compute_similarity(self, embedding_generator):
        """Test similarity computation"""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]
        
        # Identical embeddings should have similarity 1.0
        sim1 = embedding_generator.compute_similarity(emb1, emb2)
        assert abs(sim1 - 1.0) < 0.001
        
        # Orthogonal embeddings should have similarity 0.0
        sim2 = embedding_generator.compute_similarity(emb1, emb3)
        assert abs(sim2 - 0.0) < 0.001
    
    def test_preprocess_text(self, embedding_generator):
        """Test text preprocessing"""
        # Test basic cleaning
        text = "  Test   text   with   spaces  "
        processed = embedding_generator._preprocess_text(text)
        assert processed == "Test text with spaces"
        
        # Test empty text
        processed = embedding_generator._preprocess_text("")
        assert processed == ""
        
        # Test None input
        processed = embedding_generator._preprocess_text(None)
        assert processed == ""


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_no_wait(self):
        """Test rate limiter when no waiting is needed"""
        limiter = RateLimiter(max_requests_per_minute=60)
        
        # First request should not wait
        start_time = asyncio.get_event_loop().time()
        await limiter.wait_if_needed()
        end_time = asyncio.get_event_loop().time()
        
        # Should be very fast (no waiting)
        assert (end_time - start_time) < 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_tracking(self):
        """Test that rate limiter tracks requests"""
        limiter = RateLimiter(max_requests_per_minute=2)
        
        # Make requests up to the limit
        await limiter.wait_if_needed()
        await limiter.wait_if_needed()
        
        # Should have 2 requests tracked
        assert len(limiter.requests) == 2


class TestGeminiClient:
    """Test Gemini API client (mocked)"""
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock Gemini client"""
        with patch('video_rag.gemini_client.genai') as mock_genai:
            # Mock the model and its methods
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "Mock response from Gemini"
            mock_model.generate_content.return_value = mock_response
            
            mock_genai.GenerativeModel.return_value = mock_model
            
            client = GeminiClient()
            return client
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, mock_gemini_client):
        """Test summary generation"""
        transcript = "This is a test transcript about CMake basics."
        summary = await mock_gemini_client.generate_summary(transcript)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_extract_key_points(self, mock_gemini_client):
        """Test key point extraction"""
        content = "CMake is a build system. It uses CMakeLists.txt files. You can create targets."
        
        with patch.object(mock_gemini_client, '_generate_content_async') as mock_generate:
            mock_response = Mock()
            mock_response.text = "1. CMake is a build system\n2. Uses CMakeLists.txt files\n3. Can create targets"
            mock_generate.return_value = mock_response
            
            key_points = await mock_gemini_client.extract_key_points(content)
            
            assert isinstance(key_points, list)
            assert len(key_points) > 0
    
    @pytest.mark.asyncio
    async def test_answer_query(self, mock_gemini_client):
        """Test query answering"""
        query = "What is CMake?"
        context = ["CMake is a build system", "It generates build files"]
        
        answer = await mock_gemini_client.answer_query(query, context)
        
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_mock(self):
        """Test end-to-end workflow with mocks"""
        # This test verifies that all components can work together
        # even if we can't test with real video files
        
        with patch('video_rag.config.validate_config', return_value=True):
            with patch('video_rag.rag_engine.VideoProcessor') as mock_processor:
                with patch('video_rag.rag_engine.VectorStore') as mock_vector_store:
                    with patch('video_rag.rag_engine.GeminiClient') as mock_gemini:
                        
                        # Mock the video processor
                        mock_result = ProcessingResult(
                            video_id="test_id",
                            status=ProcessingStatus.COMPLETED,
                            transcript="Test transcript",
                            summary="Test summary",
                            processing_time=10.0
                        )
                        mock_processor.return_value.process_video = AsyncMock(return_value=mock_result)
                        
                        # Mock the vector store
                        mock_vector_store.return_value.store_video_content = AsyncMock()
                        mock_vector_store.return_value.search_similar_content = AsyncMock(return_value=[])
                        
                        # Mock the Gemini client
                        mock_gemini.return_value.answer_query = AsyncMock(return_value="Mock answer")
                        
                        # Import and test the RAG engine
                        from video_rag.rag_engine import VideoRAGEngine
                        
                        rag = VideoRAGEngine()
                        
                        # Test video processing
                        video_id = await rag.process_video("fake_video.mp4")
                        assert video_id == "test_id"
                        
                        # Test querying
                        result = await rag.query("Test query")
                        assert isinstance(result, QueryResult)
                        assert result.query == "Test query"


def test_imports():
    """Test that all modules can be imported"""
    try:
        from video_rag import VideoRAGEngine
        from video_rag.config import Config
        from video_rag.core.models import VideoMetadata
        from video_rag.core.exceptions import VideoRAGException
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

