cmake_minimum_required(VERSION 3.18)

project(homework_multilevel LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(multilevel_lp
    src/multilevel_lp.cu
    src/graph.cpp
)

target_include_directories(multilevel_lp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(multilevel_lp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>
)

set_target_properties(multilevel_lp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
