"""
Video processing pipeline for extracting content and metadata
"""

import os
import uuid
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
import ffmpeg
from moviepy.editor import VideoFileClip
import whisper
import numpy as np
from PIL import Image

from .config import get_config
from .core.models import VideoMetadata, ProcessingResult, ProcessingStatus
from .core.exceptions import ProcessingError, VideoFormatError
from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing and content extraction"""
    
    def __init__(self):
        self.config = get_config()
        self.gemini_client = GeminiClient()
        self.whisper_model = None
        self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Load Whisper model for transcription"""
        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            self.whisper_model = None
    
    async def process_video(self, video_path: str, metadata: Dict = None) -> ProcessingResult:
        """
        Process a video file and extract all relevant information
        
        Args:
            video_path: Path to the video file
            metadata: Optional metadata dictionary
            
        Returns:
            ProcessingResult with extracted information
        """
        start_time = asyncio.get_event_loop().time()
        video_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting video processing for {video_path}")
            
            # Validate video file
            self._validate_video_file(video_path)
            
            # Extract basic metadata
            video_metadata = await self._extract_video_metadata(video_path, video_id, metadata)
            
            # Extract audio and generate transcript
            transcript = await self._extract_transcript(video_path)
            
            # Extract key frames
            key_frames = await self._extract_key_frames(video_path)
            
            # Analyze with Gemini
            gemini_analysis = await self._analyze_with_gemini(video_path, transcript)
            
            # Generate summary and key points
            summary = await self._generate_summary(transcript, gemini_analysis)
            key_points = await self._extract_key_points(transcript, gemini_analysis)
            
            # Extract code examples
            code_examples = self._extract_code_examples(transcript)
            
            # Create processing result
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = ProcessingResult(
                video_id=video_id,
                status=ProcessingStatus.COMPLETED,
                transcript=transcript,
                summary=summary,
                key_points=key_points,
                extracted_code=code_examples,
                processing_time=processing_time,
                gemini_analysis=gemini_analysis,
                visual_elements=key_frames,
                technical_concepts=self._extract_technical_concepts(transcript, gemini_analysis)
            )
            
            logger.info(f"Video processing completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Video processing failed: {str(e)}")
            
            return ProcessingResult(
                video_id=video_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _validate_video_file(self, video_path: str):
        """Validate video file format and size"""
        if not os.path.exists(video_path):
            raise ProcessingError(f"Video file not found: {video_path}")
        
        # Check file extension
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in self.config.supported_video_formats:
            raise VideoFormatError(f"Unsupported video format: {file_ext}")
        
        # Check file size
        file_size = os.path.getsize(video_path)
        max_size = self.config.max_video_size_mb * 1024 * 1024
        if file_size > max_size:
            raise ProcessingError(f"Video file too large: {file_size} bytes (max: {max_size} bytes)")
    
    async def _extract_video_metadata(self, video_path: str, video_id: str, metadata: Dict = None) -> VideoMetadata:
        """Extract basic video metadata"""
        try:
            # Use moviepy to get video information
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                fps = clip.fps
                resolution = f"{clip.w}x{clip.h}"
            
            # Get file information
            file_size = os.path.getsize(video_path)
            filename = os.path.basename(video_path)
            file_format = Path(video_path).suffix.lower()[1:]  # Remove the dot
            
            # Create metadata object
            video_metadata = VideoMetadata(
                id=video_id,
                filename=filename,
                duration=duration,
                file_size=file_size,
                format=file_format,
                resolution=resolution,
                fps=fps,
                processing_status=ProcessingStatus.PROCESSING
            )
            
            # Add user-provided metadata
            if metadata:
                if 'title' in metadata:
                    video_metadata.title = metadata['title']
                if 'description' in metadata:
                    video_metadata.description = metadata['description']
                if 'tags' in metadata:
                    video_metadata.tags = metadata['tags']
                if 'cmake_topics' in metadata:
                    video_metadata.cmake_topics = metadata['cmake_topics']
                if 'difficulty_level' in metadata:
                    video_metadata.difficulty_level = metadata['difficulty_level']
            
            return video_metadata
            
        except Exception as e:
            raise ProcessingError(f"Failed to extract video metadata: {str(e)}")
    
    async def _extract_transcript(self, video_path: str) -> Optional[str]:
        """Extract audio and generate transcript using Whisper"""
        if not self.whisper_model:
            logger.warning("Whisper model not available, skipping transcription")
            return None
        
        try:
            logger.info("Extracting audio and generating transcript")
            
            # Extract audio using moviepy
            temp_audio_path = f"/tmp/audio_{uuid.uuid4()}.wav"
            
            with VideoFileClip(video_path) as clip:
                if clip.audio is not None:
                    clip.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                else:
                    logger.warning("No audio track found in video")
                    return None
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_audio_path)
            transcript = result["text"]
            
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            
            logger.info(f"Transcript generated: {len(transcript)} characters")
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return None
    
    async def _extract_key_frames(self, video_path: str) -> List[str]:
        """Extract key frames from video for visual analysis"""
        try:
            logger.info("Extracting key frames")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ProcessingError("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Extract frames at specified intervals
            interval = self.config.frame_extraction_interval
            frame_times = np.arange(0, duration, interval)
            
            key_frames = []
            for frame_time in frame_times[:10]:  # Limit to 10 frames
                frame_number = int(frame_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret:
                    # Save frame as base64 string or description
                    frame_description = f"Frame at {frame_time:.1f}s"
                    key_frames.append(frame_description)
            
            cap.release()
            logger.info(f"Extracted {len(key_frames)} key frames")
            return key_frames
            
        except Exception as e:
            logger.error(f"Key frame extraction failed: {str(e)}")
            return []
    
    async def _analyze_with_gemini(self, video_path: str, transcript: str) -> Optional[Dict]:
        """Analyze video with Gemini API"""
        try:
            logger.info("Analyzing video with Gemini")
            analysis = await self.gemini_client.analyze_video(video_path)
            return analysis
        except Exception as e:
            logger.error(f"Gemini analysis failed: {str(e)}")
            return None
    
    async def _generate_summary(self, transcript: str, gemini_analysis: Dict = None) -> Optional[str]:
        """Generate video summary"""
        if not transcript:
            return None
        
        try:
            context = ""
            if gemini_analysis:
                context = f"Gemini analysis: {gemini_analysis.get('summary', '')}"
            
            summary = await self.gemini_client.generate_summary(transcript, context)
            return summary
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return None
    
    async def _extract_key_points(self, transcript: str, gemini_analysis: Dict = None) -> List[str]:
        """Extract key points from video content"""
        if not transcript:
            return []
        
        try:
            content = transcript
            if gemini_analysis and 'summary' in gemini_analysis:
                content += f"\n\nAdditional context: {gemini_analysis['summary']}"
            
            key_points = await self.gemini_client.extract_key_points(content)
            return key_points
        except Exception as e:
            logger.error(f"Key point extraction failed: {str(e)}")
            return []
    
    def _extract_code_examples(self, transcript: str) -> List[str]:
        """Extract code examples from transcript"""
        if not transcript:
            return []
        
        import re
        
        # Patterns for CMake code
        cmake_patterns = [
            r'cmake_minimum_required\([^)]+\)',
            r'project\([^)]+\)',
            r'add_executable\([^)]+\)',
            r'add_library\([^)]+\)',
            r'target_link_libraries\([^)]+\)',
            r'find_package\([^)]+\)',
            r'set\([^)]+\)',
            r'target_include_directories\([^)]+\)',
        ]
        
        code_examples = []
        for pattern in cmake_patterns:
            matches = re.findall(pattern, transcript, re.IGNORECASE)
            code_examples.extend(matches)
        
        # Remove duplicates and limit
        return list(set(code_examples))[:20]
    
    def _extract_technical_concepts(self, transcript: str, gemini_analysis: Dict = None) -> List[str]:
        """Extract technical concepts mentioned in the video"""
        concepts = []
        
        # CMake-specific concepts
        cmake_concepts = [
            'cmake', 'makefile', 'build system', 'compiler', 'linker',
            'static library', 'shared library', 'executable', 'target',
            'dependencies', 'include directories', 'link libraries',
            'find package', 'configuration', 'generator', 'out-of-source build'
        ]
        
        if transcript:
            transcript_lower = transcript.lower()
            for concept in cmake_concepts:
                if concept in transcript_lower:
                    concepts.append(concept)
        
        # Add concepts from Gemini analysis
        if gemini_analysis and 'technical_details' in gemini_analysis:
            if isinstance(gemini_analysis['technical_details'], list):
                concepts.extend(gemini_analysis['technical_details'])
        
        return list(set(concepts))

