"""
Configuration management for Video RAG system
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config(BaseSettings):
    """Configuration settings for the Video RAG system"""
    
    # API Keys
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    google_cloud_project: Optional[str] = Field(None, env="GOOGLE_CLOUD_PROJECT")
    
    # Video Processing
    max_video_size_mb: int = Field(100, env="MAX_VIDEO_SIZE_MB")
    supported_video_formats: list = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        env="SUPPORTED_VIDEO_FORMATS"
    )
    frame_extraction_interval: int = Field(30, env="FRAME_EXTRACTION_INTERVAL")  # seconds
    
    # Vector Database
    vector_db_path: str = Field("./data/chroma_db", env="VECTOR_DB_PATH")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    max_chunk_size: int = Field(1000, env="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Gemini Settings
    gemini_model: str = Field("gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_temperature: float = Field(0.7, env="GEMINI_TEMPERATURE")
    max_tokens: int = Field(8192, env="MAX_TOKENS")
    
    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Storage
    upload_dir: str = Field("./data/uploads", env="UPLOAD_DIR")
    processed_dir: str = Field("./data/processed", env="PROCESSED_DIR")
    
    # Rate Limiting
    max_requests_per_minute: int = Field(60, env="MAX_REQUESTS_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def validate_config() -> bool:
    """Validate that all required configuration is present"""
    try:
        # Check required API key
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Create necessary directories
        os.makedirs(config.upload_dir, exist_ok=True)
        os.makedirs(config.processed_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config.vector_db_path), exist_ok=True)
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

