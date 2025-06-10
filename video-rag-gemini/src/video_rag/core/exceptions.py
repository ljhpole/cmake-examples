"""
Custom exceptions for the Video RAG system
"""


class VideoRAGException(Exception):
    """Base exception for Video RAG system"""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ProcessingError(VideoRAGException):
    """Exception raised during video processing"""
    pass


class QueryError(VideoRAGException):
    """Exception raised during query processing"""
    pass


class GeminiAPIError(VideoRAGException):
    """Exception raised when Gemini API calls fail"""
    pass


class VideoFormatError(VideoRAGException):
    """Exception raised for unsupported video formats"""
    pass


class VectorStoreError(VideoRAGException):
    """Exception raised for vector database operations"""
    pass


class ConfigurationError(VideoRAGException):
    """Exception raised for configuration issues"""
    pass

