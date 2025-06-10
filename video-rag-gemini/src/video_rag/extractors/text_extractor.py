"""
Text content extractor for video transcripts and descriptions
"""

import re
from typing import List, Dict, Any


class TextExtractor:
    """Extracts and processes text content from video transcripts"""
    
    def __init__(self):
        self.cmake_keywords = [
            'cmake_minimum_required', 'project', 'add_executable', 'add_library',
            'target_link_libraries', 'target_include_directories', 'find_package',
            'set', 'option', 'configure_file', 'install', 'add_subdirectory',
            'target_compile_definitions', 'target_compile_options', 'add_custom_target',
            'add_custom_command', 'file', 'string', 'list', 'foreach', 'if', 'endif'
        ]
    
    def extract_cmake_commands(self, text: str) -> List[str]:
        """Extract CMake commands from text"""
        commands = []
        
        # Pattern to match CMake function calls
        pattern = r'\b(' + '|'.join(self.cmake_keywords) + r')\s*\([^)]*\)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Get the full command including parameters
            full_match = re.search(
                rf'\b{re.escape(match)}\s*\([^)]*\)', 
                text, 
                re.IGNORECASE
            )
            if full_match:
                commands.append(full_match.group(0))
        
        return list(set(commands))  # Remove duplicates
    
    def extract_file_paths(self, text: str) -> List[str]:
        """Extract file paths mentioned in text"""
        # Common file extensions in CMake projects
        extensions = ['.cpp', '.c', '.h', '.hpp', '.cmake', '.txt', '.so', '.a', '.lib']
        
        paths = []
        for ext in extensions:
            pattern = rf'\S+{re.escape(ext)}\b'
            matches = re.findall(pattern, text, re.IGNORECASE)
            paths.extend(matches)
        
        return list(set(paths))
    
    def extract_variable_names(self, text: str) -> List[str]:
        """Extract CMake variable names from text"""
        # Pattern for ${VARIABLE_NAME} or $ENV{VARIABLE_NAME}
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, text)
        
        # Also look for SET(VARIABLE_NAME ...) patterns
        set_pattern = r'\bset\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)'
        set_matches = re.findall(set_pattern, text, re.IGNORECASE)
        
        all_variables = matches + set_matches
        return list(set(all_variables))
    
    def extract_target_names(self, text: str) -> List[str]:
        """Extract target names from CMake commands"""
        targets = []
        
        # Patterns for different target-creating commands
        patterns = [
            r'add_executable\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
            r'add_library\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)',
            r'add_custom_target\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            targets.extend(matches)
        
        return list(set(targets))
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key CMake concepts mentioned in text"""
        concepts = []
        
        concept_patterns = {
            'build system': r'\bbuild\s+system\b',
            'static library': r'\bstatic\s+librar(y|ies)\b',
            'shared library': r'\bshared\s+librar(y|ies)\b',
            'executable': r'\bexecutable\b',
            'target': r'\btarget\b',
            'dependency': r'\bdependenc(y|ies)\b',
            'linking': r'\blink(ing|ed)?\b',
            'compilation': r'\bcompil(e|ation|ing)\b',
            'configuration': r'\bconfigur(e|ation|ing)\b',
            'generator': r'\bgenerator\b',
            'cross-platform': r'\bcross[- ]platform\b',
            'out-of-source': r'\bout[- ]of[- ]source\b',
            'in-source': r'\bin[- ]source\b'
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                concepts.append(concept)
        
        return concepts
    
    def clean_transcript(self, transcript: str) -> str:
        """Clean and normalize transcript text"""
        if not transcript:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', transcript)
        
        # Remove filler words common in speech
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so basically']
        for filler in filler_words:
            pattern = rf'\b{re.escape(filler)}\b'
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def extract_timestamps_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Extract timestamp mentions from text"""
        # Pattern for time mentions like "at 2:30" or "around 5 minutes"
        time_patterns = [
            r'at\s+(\d+):(\d+)',
            r'around\s+(\d+)\s+minutes?',
            r'(\d+)\s+seconds?\s+in',
            r'timestamp\s+(\d+):(\d+)'
        ]
        
        timestamps = []
        for pattern in time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                timestamps.append({
                    'text': match.group(0),
                    'position': match.span(),
                    'groups': match.groups()
                })
        
        return timestamps
    
    def extract_questions_and_answers(self, text: str) -> List[Dict[str, str]]:
        """Extract Q&A pairs from text"""
        qa_pairs = []
        
        # Simple pattern for Q&A detection
        lines = text.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            
            # Check if line starts with question indicators
            if re.match(r'^(Q:|Question:|How|What|Why|When|Where|Can|Should|Is|Are|Do|Does)', line, re.IGNORECASE):
                current_question = line
            
            # Check if line starts with answer indicators
            elif re.match(r'^(A:|Answer:|So|Well|The answer)', line, re.IGNORECASE) and current_question:
                qa_pairs.append({
                    'question': current_question,
                    'answer': line
                })
                current_question = None
        
        return qa_pairs
    
    def extract_all_content(self, transcript: str) -> Dict[str, Any]:
        """Extract all types of content from transcript"""
        cleaned_transcript = self.clean_transcript(transcript)
        
        return {
            'cleaned_transcript': cleaned_transcript,
            'cmake_commands': self.extract_cmake_commands(cleaned_transcript),
            'file_paths': self.extract_file_paths(cleaned_transcript),
            'variable_names': self.extract_variable_names(cleaned_transcript),
            'target_names': self.extract_target_names(cleaned_transcript),
            'key_concepts': self.extract_key_concepts(cleaned_transcript),
            'timestamps': self.extract_timestamps_mentions(cleaned_transcript),
            'qa_pairs': self.extract_questions_and_answers(cleaned_transcript)
        }

