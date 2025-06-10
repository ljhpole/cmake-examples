"""
Gemini API client for video analysis and text generation
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging

from .config import get_config
from .core.exceptions import GeminiAPIError

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google's Gemini API"""
    
    def __init__(self):
        self.config = get_config()
        self._setup_client()
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute)
    
    def _setup_client(self):
        """Initialize the Gemini client"""
        try:
            genai.configure(api_key=self.config.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name=self.config.gemini_model,
                generation_config={
                    "temperature": self.config.gemini_temperature,
                    "max_output_tokens": self.config.max_tokens,
                }
            )
            logger.info(f"Gemini client initialized with model: {self.config.gemini_model}")
        except Exception as e:
            raise GeminiAPIError(f"Failed to initialize Gemini client: {str(e)}")
    
    async def analyze_video(self, video_path: str, prompt: str = None) -> Dict[str, Any]:
        """
        Analyze a video using Gemini's multimodal capabilities
        
        Args:
            video_path: Path to the video file
            prompt: Custom prompt for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        await self.rate_limiter.wait_if_needed()
        
        try:
            # Default prompt for CMake tutorial analysis
            if prompt is None:
                prompt = self._get_default_video_analysis_prompt()
            
            # Upload video file
            video_file = genai.upload_file(path=video_path)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise GeminiAPIError(f"Video processing failed: {video_file.state}")
            
            # Generate analysis
            response = await self._generate_content_async([video_file, prompt])
            
            # Parse the response
            analysis = self._parse_video_analysis(response.text)
            
            # Clean up uploaded file
            genai.delete_file(video_file.name)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            raise GeminiAPIError(f"Video analysis failed: {str(e)}")
    
    async def generate_summary(self, transcript: str, context: str = "") -> str:
        """
        Generate a summary of video content
        
        Args:
            transcript: Video transcript
            context: Additional context about the video
            
        Returns:
            Generated summary
        """
        await self.rate_limiter.wait_if_needed()
        
        prompt = f"""
        Please create a comprehensive summary of this CMake tutorial video based on the transcript below.
        
        Context: {context}
        
        Transcript:
        {transcript}
        
        Please provide:
        1. A brief overview (2-3 sentences)
        2. Key concepts covered
        3. Important code examples or commands
        4. Learning objectives
        5. Prerequisites or difficulty level
        
        Format the response as structured text with clear sections.
        """
        
        try:
            response = await self._generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            raise GeminiAPIError(f"Summary generation failed: {str(e)}")
    
    async def extract_key_points(self, content: str) -> List[str]:
        """
        Extract key points from video content
        
        Args:
            content: Video content (transcript, description, etc.)
            
        Returns:
            List of key points
        """
        await self.rate_limiter.wait_if_needed()
        
        prompt = f"""
        Extract the most important key points from this CMake tutorial content.
        Focus on actionable information, important concepts, and practical tips.
        
        Content:
        {content}
        
        Please provide 5-10 key points as a numbered list.
        Each point should be concise but informative.
        """
        
        try:
            response = await self._generate_content_async(prompt)
            # Parse the numbered list
            key_points = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and clean up
                    point = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if point:
                        key_points.append(point)
            
            return key_points
        except Exception as e:
            logger.error(f"Key point extraction failed: {str(e)}")
            raise GeminiAPIError(f"Key point extraction failed: {str(e)}")
    
    async def answer_query(self, query: str, context: List[str]) -> str:
        """
        Answer a user query using retrieved context
        
        Args:
            query: User's question
            context: List of relevant context chunks
            
        Returns:
            Generated answer
        """
        await self.rate_limiter.wait_if_needed()
        
        context_text = "\n\n".join(context)
        
        prompt = f"""
        You are an expert CMake assistant. Answer the user's question based on the provided context from CMake tutorial videos.
        
        Context from videos:
        {context_text}
        
        User Question: {query}
        
        Please provide a comprehensive answer that:
        1. Directly addresses the question
        2. References specific information from the context
        3. Includes relevant code examples if applicable
        4. Suggests next steps or related topics if helpful
        
        If the context doesn't contain enough information to fully answer the question, say so and provide what information you can.
        """
        
        try:
            response = await self._generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Query answering failed: {str(e)}")
            raise GeminiAPIError(f"Query answering failed: {str(e)}")
    
    async def _generate_content_async(self, prompt) -> Any:
        """Generate content asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.model.generate_content(
                prompt,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
        )
    
    def _get_default_video_analysis_prompt(self) -> str:
        """Get the default prompt for video analysis"""
        return """
        Analyze this CMake tutorial video and provide a detailed analysis including:
        
        1. **Content Summary**: What is this video about? What CMake concepts are being taught?
        
        2. **Technical Details**: 
           - CMake commands and functions demonstrated
           - Code examples shown
           - File structures or project layouts
           - Build configurations discussed
        
        3. **Visual Elements**:
           - Screenshots of code editors or IDEs
           - Terminal/command line demonstrations
           - Diagrams or visual aids
           - File browser or project structure views
        
        4. **Learning Objectives**: What should viewers learn from this video?
        
        5. **Difficulty Level**: Beginner, Intermediate, or Advanced?
        
        6. **Key Timestamps**: Important moments or sections (if identifiable)
        
        7. **Prerequisites**: What knowledge is assumed?
        
        8. **Related Topics**: What other CMake topics does this relate to?
        
        Please provide the analysis in a structured JSON format with clear sections.
        """
    
    def _parse_video_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse the video analysis response"""
        try:
            # Try to extract JSON if present
            import json
            import re
            
            # Look for JSON blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no JSON, parse as structured text
            analysis = {
                "summary": "",
                "technical_details": [],
                "visual_elements": [],
                "learning_objectives": [],
                "difficulty_level": "Unknown",
                "key_timestamps": [],
                "prerequisites": [],
                "related_topics": [],
                "raw_response": response_text
            }
            
            # Basic parsing of structured text
            sections = response_text.split('\n\n')
            for section in sections:
                if 'summary' in section.lower():
                    analysis["summary"] = section
                elif 'difficulty' in section.lower():
                    if 'beginner' in section.lower():
                        analysis["difficulty_level"] = "Beginner"
                    elif 'intermediate' in section.lower():
                        analysis["difficulty_level"] = "Intermediate"
                    elif 'advanced' in section.lower():
                        analysis["difficulty_level"] = "Advanced"
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Failed to parse video analysis: {str(e)}")
            return {
                "summary": response_text[:500] + "..." if len(response_text) > 500 else response_text,
                "raw_response": response_text,
                "parse_error": str(e)
            }


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        # Check if we need to wait
        if len(self.requests) >= self.max_requests:
            wait_time = 60 - (now - self.requests[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.requests.append(now)

