#!/usr/bin/env python3
"""
Example queries for the Video RAG system
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_rag import VideoRAGEngine
from video_rag.config import validate_config


class QueryExamples:
    """Collection of example queries for CMake video content"""
    
    def __init__(self):
        self.rag = None
    
    async def initialize(self):
        """Initialize the RAG engine"""
        if not validate_config():
            raise RuntimeError("Configuration validation failed")
        
        self.rag = VideoRAGEngine()
        print("✅ Video RAG Engine initialized")
    
    async def run_basic_queries(self):
        """Run basic CMake queries"""
        
        print("\n🔍 Basic CMake Queries")
        print("=" * 40)
        
        basic_queries = [
            "What is CMake?",
            "How do I create a simple CMake project?",
            "What is the difference between add_executable and add_library?",
            "How do I set the minimum CMake version?",
            "What is a CMakeLists.txt file?"
        ]
        
        for query in basic_queries:
            await self._execute_query(query, "Basic")
    
    async def run_intermediate_queries(self):
        """Run intermediate CMake queries"""
        
        print("\n🔧 Intermediate CMake Queries")
        print("=" * 40)
        
        intermediate_queries = [
            "How do I link external libraries in CMake?",
            "What are CMake targets and how do I use them?",
            "How do I set include directories for a target?",
            "What is the difference between PUBLIC, PRIVATE, and INTERFACE?",
            "How do I use find_package in CMake?",
            "How to create and use static libraries?",
            "What are CMake variables and how do I set them?"
        ]
        
        for query in intermediate_queries:
            await self._execute_query(query, "Intermediate")
    
    async def run_advanced_queries(self):
        """Run advanced CMake queries"""
        
        print("\n🚀 Advanced CMake Queries")
        print("=" * 40)
        
        advanced_queries = [
            "How do I create custom CMake functions and macros?",
            "What are CMake generators and how do I use them?",
            "How to implement cross-platform builds with CMake?",
            "How do I use CMake with external package managers?",
            "What are the best practices for large CMake projects?",
            "How to debug CMake configuration issues?",
            "How do I create installable CMake packages?"
        ]
        
        for query in advanced_queries:
            await self._execute_query(query, "Advanced")
    
    async def run_specific_use_cases(self):
        """Run queries for specific use cases"""
        
        print("\n🎯 Specific Use Case Queries")
        print("=" * 40)
        
        use_case_queries = [
            "How do I build a C++ project with CMake?",
            "How to integrate Qt with CMake?",
            "How do I use CMake with Visual Studio?",
            "How to set up unit testing with CMake and CTest?",
            "How do I create a CMake project for embedded systems?",
            "How to use CMake with Docker?",
            "How do I migrate from Makefile to CMake?"
        ]
        
        for query in use_case_queries:
            await self._execute_query(query, "Use Case")
    
    async def run_troubleshooting_queries(self):
        """Run troubleshooting-related queries"""
        
        print("\n🔧 Troubleshooting Queries")
        print("=" * 40)
        
        troubleshooting_queries = [
            "Why is my CMake build failing?",
            "How do I fix CMake linking errors?",
            "What does 'target not found' error mean in CMake?",
            "How to resolve CMake configuration warnings?",
            "Why can't CMake find my library?",
            "How to debug CMake variable values?",
            "What are common CMake mistakes to avoid?"
        ]
        
        for query in troubleshooting_queries:
            await self._execute_query(query, "Troubleshooting")
    
    async def run_filtered_queries(self):
        """Run queries with topic filters"""
        
        print("\n🏷️ Filtered Queries (by Topic)")
        print("=" * 40)
        
        filtered_queries = [
            {
                "query": "How do I work with libraries?",
                "topics": ["library", "linking"],
                "description": "Library-related content"
            },
            {
                "query": "How do I configure build settings?",
                "topics": ["build", "configuration"],
                "description": "Build configuration content"
            },
            {
                "query": "How do I manage dependencies?",
                "topics": ["dependencies", "packages"],
                "description": "Dependency management content"
            }
        ]
        
        for item in filtered_queries:
            print(f"\n🔍 Query: {item['query']}")
            print(f"🏷️ Filter: {item['description']} ({', '.join(item['topics'])})")
            
            try:
                result = await self.rag.query(
                    query_text=item['query'],
                    max_results=3,
                    filter_by_topics=item['topics']
                )
                
                self._print_result(result, "Filtered")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
    
    async def run_batch_queries(self):
        """Run multiple queries in batch"""
        
        print("\n📦 Batch Query Example")
        print("=" * 40)
        
        batch_queries = [
            "What is CMake?",
            "How to create executables?",
            "How to link libraries?",
            "What are CMake best practices?"
        ]
        
        print(f"🔄 Processing {len(batch_queries)} queries in batch...")
        
        try:
            # Simulate batch processing
            results = []
            for query in batch_queries:
                result = await self.rag.query(query_text=query, max_results=2)
                results.append(result)
            
            print(f"✅ Batch processing completed!")
            print(f"📊 Results summary:")
            
            for i, (query, result) in enumerate(zip(batch_queries, results), 1):
                print(f"   {i}. {query}")
                print(f"      Confidence: {result.confidence_score:.2%}")
                print(f"      Sources: {len(result.sources)}")
                print(f"      Time: {result.processing_time:.2f}s")
        
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    async def demonstrate_suggestions(self):
        """Demonstrate query suggestions"""
        
        print("\n💡 Query Suggestions Demo")
        print("=" * 40)
        
        base_queries = [
            "CMake basics",
            "library linking",
            "build configuration",
            "project structure"
        ]
        
        for base_query in base_queries:
            print(f"\n🔍 Base query: '{base_query}'")
            try:
                suggestions = await self.rag.suggest_related_queries(base_query)
                print("💡 Suggested related queries:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"   {i}. {suggestion}")
            except Exception as e:
                print(f"❌ Failed to get suggestions: {e}")
    
    async def _execute_query(self, query: str, category: str):
        """Execute a single query and display results"""
        
        print(f"\n🔍 [{category}] {query}")
        
        try:
            result = await self.rag.query(
                query_text=query,
                max_results=3,
                min_confidence=0.1
            )
            
            self._print_result(result, category)
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    def _print_result(self, result, category: str):
        """Print query result in a formatted way"""
        
        # Truncate response for readability
        response = result.response
        if len(response) > 200:
            response = response[:200] + "..."
        
        print(f"🤖 Response: {response}")
        print(f"📊 Confidence: {result.confidence_score:.2%}")
        print(f"⏱️ Time: {result.processing_time:.2f}s")
        
        if result.sources:
            print(f"📚 Sources: {len(result.sources)} found")
            for i, source in enumerate(result.sources[:2], 1):
                content_preview = source.get('content_preview', '')
                if len(content_preview) > 100:
                    content_preview = content_preview[:100] + "..."
                print(f"   {i}. {source['content_type']} ({source['similarity_score']:.2%})")
                print(f"      {content_preview}")
        else:
            print("📚 No sources found")
    
    async def run_all_examples(self):
        """Run all query examples"""
        
        print("🎬 Starting Video RAG Query Examples")
        print("=" * 50)
        
        try:
            await self.initialize()
            
            # Check if we have any videos to query
            stats = self.rag.get_stats()
            if stats.get('total_videos', 0) == 0:
                print("\n⚠️ No videos found in the database!")
                print("Please process some videos first using:")
                print("   python examples/process_cmake_video.py")
                print("\nRunning demo queries anyway (may return empty results)...")
            
            # Run different categories of queries
            await self.run_basic_queries()
            await self.run_intermediate_queries()
            await self.run_advanced_queries()
            await self.run_specific_use_cases()
            await self.run_troubleshooting_queries()
            await self.run_filtered_queries()
            await self.run_batch_queries()
            await self.demonstrate_suggestions()
            
            print("\n✅ All query examples completed!")
            
        except Exception as e:
            print(f"❌ Failed to run examples: {e}")


async def interactive_query_mode():
    """Interactive mode for testing custom queries"""
    
    print("🎯 Interactive Query Mode")
    print("Type your CMake questions (or 'quit' to exit)")
    print("-" * 50)
    
    try:
        examples = QueryExamples()
        await examples.initialize()
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Parse optional parameters
                max_results = 5
                topics = []
                
                if " --max=" in query:
                    parts = query.split(" --max=")
                    query = parts[0].strip()
                    try:
                        max_results = int(parts[1].split()[0])
                    except:
                        pass
                
                if " --topics=" in query:
                    parts = query.split(" --topics=")
                    query = parts[0].strip()
                    try:
                        topics = [t.strip() for t in parts[1].split()[0].split(',')]
                    except:
                        pass
                
                print("🔍 Searching...")
                result = await examples.rag.query(
                    query_text=query,
                    max_results=max_results,
                    filter_by_topics=topics if topics else None
                )
                
                examples._print_result(result, "Interactive")
                
                # Show suggestions
                suggestions = await examples.rag.suggest_related_queries(query)
                if suggestions:
                    print("\n💡 Related suggestions:")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        print(f"   {i}. {suggestion}")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to start interactive mode: {e}")


def main():
    """Main function"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_query_mode())
    else:
        examples = QueryExamples()
        asyncio.run(examples.run_all_examples())


if __name__ == "__main__":
    main()

