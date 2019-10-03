# Necessary to set CMake minimum version (3.1 needed for Spike):
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# Spike needs C++11:
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Don't want to build Spike examples:
set(BUILD_EXAMPLES OFF CACHE BOOL "Do not build Spike examples")

# Need to build Spike from the submodule:
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/Spike ${CMAKE_CURRENT_BINARY_DIR}/Spike/Build)

# Make Spike headers available:
find_package(CUDA REQUIRED)
include_directories(BEFORE SYSTEM "${CUDA_INCLUDE_DIRS}")
include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/Spike")

# Add List of Executables
foreach(model
	BinaryNetwork
    )
  add_executable(${model} ${model}.cpp)
  target_link_libraries(${model} Spike
  ${CUDA_LIBRARIES})
endforeach()