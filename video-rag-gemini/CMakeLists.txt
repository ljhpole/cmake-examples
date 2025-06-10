# Video RAG System with Gemini API
# CMake configuration for integration with cmake-examples repository

cmake_minimum_required(VERSION 3.16)
project(VideoRAGGemini VERSION 1.0.0 LANGUAGES NONE)

# Set Python requirements
set(PYTHON_MIN_VERSION 3.8)

# Find Python
find_package(Python3 ${PYTHON_MIN_VERSION} REQUIRED COMPONENTS Interpreter)

# Define installation directories
set(VIDEO_RAG_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/video-rag-gemini)
set(VIDEO_RAG_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share/video-rag-gemini)

# Custom target to install Python dependencies
add_custom_target(install_python_deps
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    COMMENT "Installing Python dependencies"
    VERBATIM
)

# Custom target to setup the environment
add_custom_target(setup_environment
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data/uploads
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data/processed
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data/chroma_db
    COMMENT "Setting up Video RAG environment"
    VERBATIM
)

# Custom target to run the API server
add_custom_target(run_server
    COMMAND ${Python3_EXECUTABLE} -m uvicorn src.video_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Starting Video RAG API server"
    VERBATIM
    DEPENDS setup_environment
)

# Custom target to run tests
add_custom_target(run_tests
    COMMAND ${Python3_EXECUTABLE} -m pytest tests/ -v
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running Video RAG tests"
    VERBATIM
)

# Custom target to process example video
add_custom_target(process_example
    COMMAND ${Python3_EXECUTABLE} examples/process_cmake_video.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Processing example CMake video"
    VERBATIM
    DEPENDS setup_environment
)

# Installation rules
install(DIRECTORY src/
    DESTINATION ${VIDEO_RAG_INSTALL_DIR}/src
    FILES_MATCHING PATTERN "*.py"
)

install(FILES requirements.txt
    DESTINATION ${VIDEO_RAG_INSTALL_DIR}
)

install(DIRECTORY examples/
    DESTINATION ${VIDEO_RAG_INSTALL_DIR}/examples
    FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY docs/
    DESTINATION ${VIDEO_RAG_DATA_DIR}/docs
    FILES_MATCHING PATTERN "*.md"
)

install(FILES README.md
    DESTINATION ${VIDEO_RAG_INSTALL_DIR}
)

# Create .env template
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/.env.template
    ${CMAKE_BINARY_DIR}/.env.example
    COPYONLY
)

install(FILES ${CMAKE_BINARY_DIR}/.env.example
    DESTINATION ${VIDEO_RAG_INSTALL_DIR}
)

# Print configuration summary
message(STATUS "Video RAG System Configuration:")
message(STATUS "  Python version: ${Python3_VERSION}")
message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
message(STATUS "  Install directory: ${VIDEO_RAG_INSTALL_DIR}")
message(STATUS "  Data directory: ${VIDEO_RAG_DATA_DIR}")
message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  install_python_deps - Install Python dependencies")
message(STATUS "  setup_environment   - Setup data directories")
message(STATUS "  run_server          - Start the API server")
message(STATUS "  run_tests           - Run the test suite")
message(STATUS "  process_example     - Process example video")
message(STATUS "")
message(STATUS "Quick start:")
message(STATUS "  1. make install_python_deps")
message(STATUS "  2. Copy .env.example to .env and configure")
message(STATUS "  3. make run_server")

# Add dependencies
add_dependencies(run_server install_python_deps)
add_dependencies(run_tests install_python_deps)
add_dependencies(process_example install_python_deps)

