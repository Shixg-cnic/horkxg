cmake_minimum_required(VERSION 3.18)

project(single_gpu_lp_baseline LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(single_gpu_lp_baseline
    src/main.cu
    src/graph.cpp
    src/partitioner.cu
)

target_include_directories(single_gpu_lp_baseline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(single_gpu_lp_baseline PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>
)

set_target_properties(single_gpu_lp_baseline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(multilevel_lp
    src/multilevel_lp.cu
    src/graph.cpp
    src/partitioner.cu
)
target_include_directories(multilevel_lp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(multilevel_lp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>)
set_target_properties(multilevel_lp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Research references are not stages of the default partitioning algorithm.
option(BUILD_RESEARCH_TOOLS "Build experimental CPU quality references" OFF)
if(BUILD_RESEARCH_TOOLS)
add_executable(quality_refine src/quality_refine.cpp src/graph.cpp)
target_include_directories(quality_refine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(quality_refine PRIVATE -O3)

add_executable(community_seed src/community_seed.cpp src/graph.cpp)
target_include_directories(community_seed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(community_seed PRIVATE -O3)
add_executable(region_refine src/region_refine.cpp src/graph.cpp)
target_include_directories(region_refine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(region_refine PRIVATE -O3)
endif()
