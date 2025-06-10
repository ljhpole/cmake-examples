#!/usr/bin/env python3
"""
Example script for processing a CMake tutorial video
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_rag import VideoRAGEngine
from video_rag.config import validate_config


async def process_example_video():
    """Process an example CMake tutorial video"""
    
    print("🎥 CMake Video RAG Processing Example")
    print("=" * 50)
    
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed!")
        print("Please ensure you have:")
        print("1. Set GEMINI_API_KEY in your .env file")
        print("2. Installed all required dependencies")
        return
    
    # Initialize RAG engine
    print("🚀 Initializing Video RAG Engine...")
    try:
        rag = VideoRAGEngine()
        print("✅ RAG Engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG engine: {e}")
        return
    
    # Example video metadata
    video_metadata = {
        "title": "CMake Build System Basics",
        "description": "An introduction to CMake build system covering basic concepts, targets, and project structure",
        "cmake_topics": [
            "build system",
            "targets", 
            "executables",
            "libraries",
            "project structure"
        ],
        "difficulty_level": "Beginner",
        "tags": ["tutorial", "cmake", "build", "beginner"]
    }
    
    # Check if we have a sample video file
    sample_video_path = None
    possible_paths = [
        "sample_video.mp4",
        "../sample_video.mp4", 
        "examples/sample_video.mp4",
        os.path.expanduser("~/Downloads/cmake_tutorial.mp4")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sample_video_path = path
            break
    
    if not sample_video_path:
        print("📹 No sample video found. Creating a mock processing example...")
        print("\nTo test with a real video:")
        print("1. Place a CMake tutorial video in the current directory as 'sample_video.mp4'")
        print("2. Or update the 'sample_video_path' variable in this script")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .webm")
        
        # Demonstrate the API without actual video processing
        await demonstrate_api_usage(rag)
        return
    
    print(f"📹 Processing video: {sample_video_path}")
    print(f"📊 Video metadata: {video_metadata}")
    
    try:
        # Process the video
        print("\n🔄 Processing video (this may take a few minutes)...")
        video_id = await rag.process_video(sample_video_path, video_metadata)
        print(f"✅ Video processed successfully!")
        print(f"📋 Video ID: {video_id}")
        
        # Get video summary
        print("\n📄 Getting video summary...")
        summary = await rag.get_video_summary(video_id)
        if summary:
            print("📋 Video Summary:")
            print(f"   Title: {summary.get('metadata', {}).get('title', 'N/A')}")
            print(f"   Duration: {summary.get('metadata', {}).get('duration', 'N/A')} seconds")
            print(f"   Topics: {summary.get('metadata', {}).get('cmake_topics', [])}")
            if 'summary' in summary:
                print(f"   Content: {summary['summary'][:200]}...")
        
        # Demonstrate querying
        await demonstrate_querying(rag, video_id)
        
    except Exception as e:
        print(f"❌ Video processing failed: {e}")
        print("This might be due to:")
        print("- Invalid video format")
        print("- Missing dependencies (FFmpeg, Whisper)")
        print("- Insufficient disk space")
        print("- Network issues with Gemini API")


async def demonstrate_api_usage(rag):
    """Demonstrate API usage without actual video processing"""
    
    print("\n🔍 API Usage Demonstration")
    print("-" * 30)
    
    # Show system stats
    print("📊 System Statistics:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Show available topics (will be empty initially)
    print("\n🏷️ Available Topics:")
    videos = await rag.list_videos()
    topics = set()
    for video in videos:
        topics.update(video.get('cmake_topics', []))
    
    if topics:
        for topic in sorted(topics):
            print(f"   - {topic}")
    else:
        print("   No topics available (no videos processed yet)")
    
    print("\n💡 Example Queries You Could Ask:")
    example_queries = [
        "How do I create a CMake project?",
        "What is the difference between add_executable and add_library?",
        "How to link libraries in CMake?",
        "What are CMake targets?",
        "How to set up include directories?",
        "Best practices for CMake project structure"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"   {i}. {query}")


async def demonstrate_querying(rag, video_id):
    """Demonstrate querying functionality"""
    
    print("\n🤖 Querying Demonstration")
    print("-" * 30)
    
    # Example queries
    queries = [
        "What is CMake?",
        "How do I create an executable?",
        "What are the basic CMake commands?",
        "How to structure a CMake project?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n❓ Query {i}: {query}")
        try:
            result = await rag.query(
                query_text=query,
                max_results=3,
                min_confidence=0.1
            )
            
            print(f"🤖 Response: {result.response[:300]}...")
            print(f"📊 Confidence: {result.confidence_score:.2%}")
            print(f"⏱️ Processing time: {result.processing_time:.2f}s")
            
            if result.sources:
                print(f"📚 Sources: {len(result.sources)} found")
                for j, source in enumerate(result.sources[:2], 1):
                    print(f"   {j}. {source['content_type']} ({source['similarity_score']:.2%} match)")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Suggest related queries
    print("\n💡 Related Query Suggestions:")
    try:
        suggestions = await rag.suggest_related_queries("CMake basics")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"   {i}. {suggestion}")
    except Exception as e:
        print(f"❌ Failed to get suggestions: {e}")


async def interactive_mode(rag):
    """Interactive query mode"""
    
    print("\n🎯 Interactive Mode")
    print("Type your questions about CMake (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print("🔍 Searching...")
            result = await rag.query(query_text=query, max_results=5)
            
            print(f"\n🤖 Answer:")
            print(result.response)
            print(f"\n📊 Confidence: {result.confidence_score:.2%}")
            
            if result.sources:
                print(f"\n📚 Sources ({len(result.sources)}):")
                for i, source in enumerate(result.sources[:3], 1):
                    print(f"   {i}. {source['content_type']} - {source['similarity_score']:.2%} match")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function"""
    
    print("🎬 Starting CMake Video RAG Example")
    
    # Check if user wants interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("🎯 Interactive mode enabled")
        asyncio.run(interactive_mode_wrapper())
    else:
        asyncio.run(process_example_video())


async def interactive_mode_wrapper():
    """Wrapper for interactive mode"""
    
    if not validate_config():
        print("❌ Configuration validation failed!")
        return
    
    try:
        rag = VideoRAGEngine()
        await interactive_mode(rag)
    except Exception as e:
        print(f"❌ Failed to start interactive mode: {e}")


if __name__ == "__main__":
    main()

