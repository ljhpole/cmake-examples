"""
Code extractor for CMake and C++ code snippets from video content
"""

import re
from typing import List, Dict, Any


class CodeExtractor:
    """Extracts code snippets from video transcripts and descriptions"""
    
    def __init__(self):
        self.cmake_functions = [
            'cmake_minimum_required', 'project', 'add_executable', 'add_library',
            'target_link_libraries', 'target_include_directories', 'find_package',
            'set', 'option', 'configure_file', 'install'
        ]
    
    def extract_cmake_snippets(self, text: str) -> List[Dict[str, Any]]:
        """Extract CMake code snippets from text"""
        snippets = []
        
        # Pattern for CMake function calls with parameters
        for func in self.cmake_functions:
            pattern = rf'\b{func}\s*\([^)]*\)'
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                snippets.append({
                    'type': 'cmake_function',
                    'function': func,
                    'code': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return snippets
    
    def extract_cpp_snippets(self, text: str) -> List[Dict[str, Any]]:
        """Extract C++ code snippets from text"""
        snippets = []
        
        # Pattern for C++ includes
        include_pattern = r'#include\s*[<"][^>"]+[>"]'
        includes = re.finditer(include_pattern, text)
        
        for match in includes:
            snippets.append({
                'type': 'cpp_include',
                'code': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        # Pattern for simple function definitions
        func_pattern = r'\b\w+\s+\w+\s*\([^)]*\)\s*\{'
        functions = re.finditer(func_pattern, text)
        
        for match in functions:
            snippets.append({
                'type': 'cpp_function',
                'code': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        return snippets
    
    def extract_command_line(self, text: str) -> List[Dict[str, Any]]:
        """Extract command line snippets"""
        snippets = []
        
        # Common command patterns
        patterns = [
            r'\$\s+cmake\s+[^\n]+',
            r'\$\s+make\s+[^\n]*',
            r'\$\s+mkdir\s+[^\n]+',
            r'\$\s+cd\s+[^\n]+',
            r'cmake\s+--build\s+[^\n]+',
            r'cmake\s+--install\s+[^\n]+'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                snippets.append({
                    'type': 'command_line',
                    'code': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return snippets
    
    def extract_file_content(self, text: str) -> List[Dict[str, Any]]:
        """Extract file content blocks"""
        snippets = []
        
        # Pattern for file content blocks (often shown in tutorials)
        # Looking for patterns like "CMakeLists.txt:" followed by content
        file_pattern = r'(\w+\.\w+):\s*\n((?:.*\n?)*?)(?=\n\w+\.\w+:|$)'
        matches = re.finditer(file_pattern, text, re.MULTILINE)
        
        for match in matches:
            filename = match.group(1)
            content = match.group(2).strip()
            
            if content:  # Only include non-empty content
                snippets.append({
                    'type': 'file_content',
                    'filename': filename,
                    'code': content,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return snippets
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract code blocks marked with common delimiters"""
        snippets = []
        
        # Markdown-style code blocks
        markdown_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.finditer(markdown_pattern, text, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'unknown'
            code = match.group(2)
            
            snippets.append({
                'type': 'code_block',
                'language': language,
                'code': code,
                'start': match.start(),
                'end': match.end()
            })
        
        # Indented code blocks (4+ spaces)
        lines = text.split('\n')
        in_code_block = False
        code_lines = []
        start_line = 0
        
        for i, line in enumerate(lines):
            if line.startswith('    ') and line.strip():  # 4+ spaces and not empty
                if not in_code_block:
                    in_code_block = True
                    start_line = i
                    code_lines = []
                code_lines.append(line[4:])  # Remove 4 spaces
            else:
                if in_code_block and code_lines:
                    snippets.append({
                        'type': 'indented_code',
                        'code': '\n'.join(code_lines),
                        'start_line': start_line,
                        'end_line': i - 1
                    })
                in_code_block = False
                code_lines = []
        
        return snippets
    
    def classify_code_snippet(self, code: str) -> str:
        """Classify the type of code snippet"""
        code_lower = code.lower()
        
        if any(func in code_lower for func in self.cmake_functions):
            return 'cmake'
        elif '#include' in code or 'int main' in code:
            return 'cpp'
        elif code.startswith('$') or 'cmake' in code_lower:
            return 'command_line'
        elif any(ext in code for ext in ['.cpp', '.h', '.cmake']):
            return 'file_reference'
        else:
            return 'unknown'
    
    def extract_all_code(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all types of code from text"""
        return {
            'cmake_snippets': self.extract_cmake_snippets(text),
            'cpp_snippets': self.extract_cpp_snippets(text),
            'command_line': self.extract_command_line(text),
            'file_content': self.extract_file_content(text),
            'code_blocks': self.extract_code_blocks(text)
        }
    
    def get_code_summary(self, text: str) -> Dict[str, Any]:
        """Get a summary of code content in text"""
        all_code = self.extract_all_code(text)
        
        total_snippets = sum(len(snippets) for snippets in all_code.values())
        
        # Count by type
        type_counts = {}
        all_snippets = []
        
        for category, snippets in all_code.items():
            type_counts[category] = len(snippets)
            all_snippets.extend(snippets)
        
        # Classify all snippets
        classifications = {}
        for snippet in all_snippets:
            code = snippet.get('code', '')
            classification = self.classify_code_snippet(code)
            classifications[classification] = classifications.get(classification, 0) + 1
        
        return {
            'total_snippets': total_snippets,
            'by_category': type_counts,
            'by_classification': classifications,
            'has_cmake': classifications.get('cmake', 0) > 0,
            'has_cpp': classifications.get('cpp', 0) > 0,
            'has_commands': classifications.get('command_line', 0) > 0
        }

