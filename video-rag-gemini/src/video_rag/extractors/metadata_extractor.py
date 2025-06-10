"""
Metadata extractor for video content analysis
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime


class MetadataExtractor:
    """Extracts metadata and structured information from video content"""
    
    def __init__(self):
        self.difficulty_indicators = {
            'beginner': ['basic', 'introduction', 'getting started', 'first', 'simple', 'easy'],
            'intermediate': ['advanced', 'complex', 'detailed', 'in-depth', 'comprehensive'],
            'advanced': ['expert', 'professional', 'optimization', 'performance', 'enterprise']
        }
        
        self.cmake_topics = {
            'basics': ['cmake', 'build system', 'makefile', 'compilation'],
            'targets': ['executable', 'library', 'target', 'add_executable', 'add_library'],
            'linking': ['link', 'target_link_libraries', 'dependencies'],
            'configuration': ['configure', 'option', 'set', 'variable'],
            'installation': ['install', 'package', 'cpack'],
            'testing': ['test', 'ctest', 'unit test'],
            'packaging': ['package', 'find_package', 'config'],
            'cross-platform': ['windows', 'linux', 'macos', 'cross-platform'],
            'advanced': ['generator', 'custom', 'macro', 'function']
        }
    
    def extract_difficulty_level(self, text: str, title: str = "") -> str:
        """Determine difficulty level from content"""
        text_lower = (text + " " + title).lower()
        
        scores = {'beginner': 0, 'intermediate': 0, 'advanced': 0}
        
        for level, indicators in self.difficulty_indicators.items():
            for indicator in indicators:
                scores[level] += text_lower.count(indicator)
        
        # Additional heuristics
        if 'introduction' in text_lower or 'getting started' in text_lower:
            scores['beginner'] += 3
        
        if len(re.findall(r'\badvanced\b|\bcomplex\b|\boptimization\b', text_lower)) > 0:
            scores['advanced'] += 2
        
        # Return the level with highest score
        max_level = max(scores, key=scores.get)
        return max_level if scores[max_level] > 0 else 'intermediate'
    
    def extract_cmake_topics(self, text: str, title: str = "") -> List[str]:
        """Extract CMake topics covered in the content"""
        content = (text + " " + title).lower()
        found_topics = []
        
        for topic, keywords in self.cmake_topics.items():
            score = 0
            for keyword in keywords:
                score += content.count(keyword)
            
            if score > 0:
                found_topics.append(topic)
        
        # Additional specific topic detection
        if 'static' in content and 'library' in content:
            found_topics.append('static libraries')
        
        if 'shared' in content and 'library' in content:
            found_topics.append('shared libraries')
        
        if 'include' in content and 'director' in content:
            found_topics.append('include directories')
        
        return list(set(found_topics))
    
    def extract_learning_objectives(self, text: str) -> List[str]:
        """Extract learning objectives from content"""
        objectives = []
        
        # Look for explicit objective statements
        objective_patterns = [
            r'you will learn[^.]*',
            r'this tutorial covers[^.]*',
            r'we will show[^.]*',
            r'by the end[^.]*',
            r'objectives?[:\s]+([^.]*)',
            r'goals?[:\s]+([^.]*)'
        ]
        
        for pattern in objective_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            objectives.extend(matches)
        
        # Look for "how to" statements
        how_to_pattern = r'how to [^.]*'
        how_to_matches = re.findall(how_to_pattern, text, re.IGNORECASE)
        objectives.extend(how_to_matches)
        
        # Clean and deduplicate
        cleaned_objectives = []
        for obj in objectives:
            obj = obj.strip().rstrip('.,')
            if len(obj) > 10 and obj not in cleaned_objectives:
                cleaned_objectives.append(obj)
        
        return cleaned_objectives[:10]  # Limit to 10 objectives
    
    def extract_prerequisites(self, text: str) -> List[str]:
        """Extract prerequisites from content"""
        prerequisites = []
        
        # Look for prerequisite statements
        prereq_patterns = [
            r'prerequisite[s]?[:\s]+([^.]*)',
            r'you should know[^.]*',
            r'before starting[^.]*',
            r'assumes? knowledge of[^.]*',
            r'requires?[^.]*knowledge',
            r'familiarity with[^.]*'
        ]
        
        for pattern in prereq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            prerequisites.extend(matches)
        
        # Common CMake prerequisites
        if 'c++' in text.lower() or 'cpp' in text.lower():
            prerequisites.append('Basic C++ knowledge')
        
        if 'command line' in text.lower() or 'terminal' in text.lower():
            prerequisites.append('Command line familiarity')
        
        # Clean and deduplicate
        cleaned_prereqs = []
        for prereq in prerequisites:
            prereq = prereq.strip().rstrip('.,')
            if len(prereq) > 5 and prereq not in cleaned_prereqs:
                cleaned_prereqs.append(prereq)
        
        return cleaned_prereqs[:5]  # Limit to 5 prerequisites
    
    def extract_duration_estimate(self, text: str, actual_duration: float = None) -> Dict[str, Any]:
        """Extract or estimate content duration information"""
        duration_info = {
            'actual_duration': actual_duration,
            'estimated_reading_time': None,
            'complexity_factor': 1.0
        }
        
        if actual_duration:
            # Categorize by duration
            if actual_duration < 300:  # 5 minutes
                duration_info['category'] = 'short'
            elif actual_duration < 1200:  # 20 minutes
                duration_info['category'] = 'medium'
            else:
                duration_info['category'] = 'long'
        
        # Estimate reading time for transcript
        word_count = len(text.split())
        if word_count > 0:
            # Average reading speed: 200 words per minute
            duration_info['estimated_reading_time'] = word_count / 200
        
        # Complexity factor based on content
        complexity_indicators = ['advanced', 'complex', 'detailed', 'comprehensive']
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text.lower())
        duration_info['complexity_factor'] = min(2.0, 1.0 + complexity_score * 0.2)
        
        return duration_info
    
    def extract_key_timestamps(self, text: str) -> List[Dict[str, Any]]:
        """Extract key timestamps and sections from content"""
        timestamps = []
        
        # Look for explicit timestamp mentions
        timestamp_patterns = [
            r'(\d{1,2}):(\d{2})\s*[-–]\s*([^.\n]+)',
            r'at\s+(\d{1,2}):(\d{2})[^a-zA-Z]*([^.\n]+)',
            r'(\d{1,2})\s+minutes?\s*[-–]\s*([^.\n]+)'
        ]
        
        for pattern in timestamp_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 3:
                    minutes, seconds, description = match.groups()
                    timestamp_seconds = int(minutes) * 60 + int(seconds)
                else:
                    minutes, description = match.groups()
                    timestamp_seconds = int(minutes) * 60
                
                timestamps.append({
                    'time_seconds': timestamp_seconds,
                    'time_formatted': f"{minutes}:{seconds if len(match.groups()) == 3 else '00'}",
                    'description': description.strip(),
                    'position_in_text': match.start()
                })
        
        return sorted(timestamps, key=lambda x: x['time_seconds'])
    
    def extract_related_topics(self, text: str, title: str = "") -> List[str]:
        """Extract related topics that might be of interest"""
        content = (text + " " + title).lower()
        related_topics = []
        
        # Build system related
        if 'cmake' in content:
            related_topics.extend(['Make', 'Ninja', 'MSBuild', 'Autotools'])
        
        # Language related
        if any(lang in content for lang in ['c++', 'cpp', 'c']):
            related_topics.extend(['C++ compilation', 'GCC', 'Clang', 'MSVC'])
        
        # Platform related
        platforms = ['windows', 'linux', 'macos', 'unix']
        found_platforms = [p for p in platforms if p in content]
        if len(found_platforms) > 1:
            related_topics.append('Cross-platform development')
        
        # Tool related
        tools = ['git', 'docker', 'jenkins', 'ci/cd', 'testing']
        found_tools = [t for t in tools if t in content]
        related_topics.extend(found_tools)
        
        return list(set(related_topics))
    
    def extract_all_metadata(self, text: str, title: str = "", 
                           actual_duration: float = None) -> Dict[str, Any]:
        """Extract all metadata from content"""
        return {
            'difficulty_level': self.extract_difficulty_level(text, title),
            'cmake_topics': self.extract_cmake_topics(text, title),
            'learning_objectives': self.extract_learning_objectives(text),
            'prerequisites': self.extract_prerequisites(text),
            'duration_info': self.extract_duration_estimate(text, actual_duration),
            'key_timestamps': self.extract_key_timestamps(text),
            'related_topics': self.extract_related_topics(text, title),
            'extraction_timestamp': datetime.utcnow().isoformat(),
            'content_stats': {
                'word_count': len(text.split()),
                'character_count': len(text),
                'line_count': len(text.split('\n'))
            }
        }

