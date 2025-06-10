"""
Video RAG System using Gemini API

A comprehensive Retrieval-Augmented Generation system for video content analysis
and intelligent querying using Google's Gemini API.
"""

__version__ = "1.0.0"
__author__ = "Codegen AI"
__description__ = "Video RAG system with Gemini API integration"

from .config import Config
from .rag_engine import VideoRAGEngine

__all__ = ["Config", "VideoRAGEngine"]

