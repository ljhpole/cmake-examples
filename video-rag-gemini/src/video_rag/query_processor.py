"""
Query processor for handling and optimizing user queries
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Processed query with extracted information"""
    original_query: str
    cleaned_query: str
    intent: str
    entities: List[str]
    cmake_topics: List[str]
    question_type: str
    complexity_level: str
    suggested_filters: Dict[str, Any]


class QueryProcessor:
    """Processes and analyzes user queries for better retrieval"""
    
    def __init__(self):
        self.cmake_entities = {
            'commands': [
                'cmake_minimum_required', 'project', 'add_executable', 'add_library',
                'target_link_libraries', 'target_include_directories', 'find_package',
                'set', 'option', 'configure_file', 'install', 'add_subdirectory'
            ],
            'concepts': [
                'target', 'library', 'executable', 'build system', 'generator',
                'configuration', 'compilation', 'linking', 'dependencies'
            ],
            'file_types': [
                'CMakeLists.txt', 'cmake', 'cpp', 'hpp', 'c', 'h'
            ]
        }
        
        self.question_patterns = {
            'how_to': [r'\bhow\s+(?:do\s+i|to|can\s+i)\b', r'\bsteps?\s+to\b'],
            'what_is': [r'\bwhat\s+is\b', r'\bwhat\s+are\b', r'\bdefine\b'],
            'why': [r'\bwhy\b', r'\breason\b', r'\bpurpose\b'],
            'when': [r'\bwhen\b', r'\bunder\s+what\b'],
            'where': [r'\bwhere\b', r'\blocation\b'],
            'comparison': [r'\bdifference\b', r'\bcompare\b', r'\bvs\b', r'\bversus\b'],
            'troubleshooting': [r'\berror\b', r'\bfail\b', r'\bproblem\b', r'\bissue\b', r'\bfix\b'],
            'best_practices': [r'\bbest\s+practice\b', r'\brecommend\b', r'\bshould\b']
        }
        
        self.intent_patterns = {
            'learning': [r'\blearn\b', r'\btutorial\b', r'\bguide\b', r'\bteach\b'],
            'implementation': [r'\bimplement\b', r'\bcreate\b', r'\bbuild\b', r'\bmake\b'],
            'debugging': [r'\bdebug\b', r'\bfix\b', r'\bsolve\b', r'\btroubleshoot\b'],
            'optimization': [r'\boptimiz\b', r'\bperformance\b', r'\befficient\b'],
            'configuration': [r'\bconfigur\b', r'\bsetup\b', r'\binstall\b']
        }
    
    def process_query(self, query: str) -> ProcessedQuery:
        """Process a user query and extract structured information"""
        
        # Clean the query
        cleaned_query = self._clean_query(query)
        
        # Extract entities
        entities = self._extract_entities(cleaned_query)
        
        # Determine question type
        question_type = self._classify_question_type(cleaned_query)
        
        # Determine intent
        intent = self._classify_intent(cleaned_query)
        
        # Extract CMake topics
        cmake_topics = self._extract_cmake_topics(cleaned_query)
        
        # Assess complexity
        complexity_level = self._assess_complexity(cleaned_query, entities)
        
        # Generate suggested filters
        suggested_filters = self._generate_filters(
            cleaned_query, entities, cmake_topics, question_type, intent
        )
        
        return ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            entities=entities,
            cmake_topics=cmake_topics,
            question_type=question_type,
            complexity_level=complexity_level,
            suggested_filters=suggested_filters
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common filler words that don't add meaning
        filler_words = ['please', 'can you', 'could you', 'i want to', 'i need to']
        for filler in filler_words:
            cleaned = re.sub(rf'\b{re.escape(filler)}\b', '', cleaned)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract CMake-related entities from the query"""
        entities = []
        
        for category, entity_list in self.cmake_entities.items():
            for entity in entity_list:
                if entity.lower() in query:
                    entities.append(entity)
        
        # Extract quoted entities
        quoted_entities = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted_entities)
        
        # Extract file extensions
        file_extensions = re.findall(r'\.\w+', query)
        entities.extend(file_extensions)
        
        return list(set(entities))
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question being asked"""
        for question_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return question_type
        
        return 'general'
    
    def _classify_intent(self, query: str) -> str:
        """Classify the user's intent"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            intent_scores[intent] = score
        
        # Return the intent with the highest score
        if intent_scores and max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        return 'general'
    
    def _extract_cmake_topics(self, query: str) -> List[str]:
        """Extract CMake topics from the query"""
        topics = []
        
        topic_keywords = {
            'basics': ['basic', 'introduction', 'getting started', 'fundamentals'],
            'targets': ['target', 'executable', 'library', 'add_executable', 'add_library'],
            'linking': ['link', 'linking', 'target_link_libraries', 'dependencies'],
            'configuration': ['configure', 'configuration', 'option', 'variable', 'set'],
            'installation': ['install', 'installation', 'package', 'cpack'],
            'testing': ['test', 'testing', 'ctest', 'unit test'],
            'cross-platform': ['cross-platform', 'windows', 'linux', 'macos', 'platform'],
            'advanced': ['advanced', 'complex', 'custom', 'macro', 'function', 'generator']
        }
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    topics.append(topic)
                    break
        
        return list(set(topics))
    
    def _assess_complexity(self, query: str, entities: List[str]) -> str:
        """Assess the complexity level of the query"""
        complexity_indicators = {
            'beginner': ['basic', 'simple', 'introduction', 'getting started', 'first'],
            'intermediate': ['configure', 'setup', 'multiple', 'different'],
            'advanced': ['advanced', 'complex', 'custom', 'optimization', 'performance']
        }
        
        scores = {'beginner': 0, 'intermediate': 0, 'advanced': 0}
        
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in query:
                    scores[level] += 1
        
        # Additional heuristics
        if len(entities) > 3:
            scores['intermediate'] += 1
        
        if any(word in query for word in ['macro', 'function', 'generator', 'cross-compile']):
            scores['advanced'] += 2
        
        # Return the level with the highest score
        max_level = max(scores, key=scores.get)
        return max_level if scores[max_level] > 0 else 'intermediate'
    
    def _generate_filters(self, query: str, entities: List[str], 
                         cmake_topics: List[str], question_type: str, 
                         intent: str) -> Dict[str, Any]:
        """Generate suggested filters for the query"""
        filters = {
            'topics': cmake_topics,
            'difficulty': None,
            'content_types': [],
            'boost_keywords': []
        }
        
        # Suggest difficulty filter based on query
        if any(word in query for word in ['beginner', 'basic', 'introduction']):
            filters['difficulty'] = 'Beginner'
        elif any(word in query for word in ['advanced', 'expert', 'complex']):
            filters['difficulty'] = 'Advanced'
        
        # Suggest content types based on question type
        if question_type == 'how_to':
            filters['content_types'].extend(['tutorial', 'guide'])
        elif question_type == 'troubleshooting':
            filters['content_types'].extend(['troubleshooting', 'debugging'])
        elif question_type == 'what_is':
            filters['content_types'].extend(['explanation', 'definition'])
        
        # Add boost keywords for better ranking
        filters['boost_keywords'] = entities + cmake_topics
        
        return filters
    
    def expand_query(self, processed_query: ProcessedQuery) -> List[str]:
        """Generate expanded queries for better retrieval"""
        expanded_queries = [processed_query.cleaned_query]
        
        # Add synonyms and related terms
        synonyms = {
            'library': ['lib', 'shared library', 'static library'],
            'executable': ['exe', 'binary', 'program'],
            'target': ['build target', 'cmake target'],
            'link': ['linking', 'linker'],
            'compile': ['compilation', 'compiler', 'build']
        }
        
        for entity in processed_query.entities:
            if entity in synonyms:
                for synonym in synonyms[entity]:
                    expanded_query = processed_query.cleaned_query.replace(entity, synonym)
                    if expanded_query != processed_query.cleaned_query:
                        expanded_queries.append(expanded_query)
        
        # Add topic-specific expansions
        for topic in processed_query.cmake_topics:
            topic_query = f"{processed_query.cleaned_query} {topic}"
            expanded_queries.append(topic_query)
        
        return expanded_queries[:5]  # Limit to 5 expanded queries
    
    def suggest_related_queries(self, processed_query: ProcessedQuery) -> List[str]:
        """Suggest related queries based on the processed query"""
        suggestions = []
        
        # Question type based suggestions
        if processed_query.question_type == 'how_to':
            suggestions.extend([
                f"What is {' '.join(processed_query.entities[:2])}?",
                f"Best practices for {' '.join(processed_query.cmake_topics[:2])}",
                f"Common errors with {' '.join(processed_query.entities[:2])}"
            ])
        elif processed_query.question_type == 'what_is':
            suggestions.extend([
                f"How to use {' '.join(processed_query.entities[:2])}?",
                f"Examples of {' '.join(processed_query.entities[:2])}",
                f"When to use {' '.join(processed_query.entities[:2])}?"
            ])
        
        # Topic-based suggestions
        for topic in processed_query.cmake_topics[:2]:
            suggestions.extend([
                f"Advanced {topic} techniques",
                f"{topic} best practices",
                f"Common {topic} mistakes"
            ])
        
        # Intent-based suggestions
        if processed_query.intent == 'learning':
            suggestions.append("CMake tutorial for beginners")
        elif processed_query.intent == 'debugging':
            suggestions.append("How to debug CMake build issues")
        
        # Remove duplicates and limit
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions and len(suggestion) > 10:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:5]
    
    def optimize_for_retrieval(self, processed_query: ProcessedQuery) -> Dict[str, Any]:
        """Optimize the query for vector retrieval"""
        return {
            'primary_query': processed_query.cleaned_query,
            'expanded_queries': self.expand_query(processed_query),
            'boost_terms': processed_query.entities + processed_query.cmake_topics,
            'filters': processed_query.suggested_filters,
            'weight_adjustments': {
                'exact_match': 1.5 if processed_query.question_type == 'what_is' else 1.0,
                'topic_match': 1.3 if processed_query.cmake_topics else 1.0,
                'entity_match': 1.2 if processed_query.entities else 1.0
            }
        }

