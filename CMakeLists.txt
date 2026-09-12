cmake_minimum_required(VERSION 3.18)

project(homework_multilevel LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# The primary test host is GH200 (SM90). Override this cache entry when using
# a different GPU, e.g. -DCMAKE_CUDA_ARCHITECTURES=80.
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(multilevel_lp
    src/main.cu
    src/sclp.cu
    src/graph.cpp
)

add_executable(sclp_diag_mr
    src/main.cu
    src/sclp.cu
    src/graph.cpp
)
target_compile_definitions(sclp_diag_mr PRIVATE SCLP_MERGE_DIAGNOSTICS=1)

add_executable(sclp_diag_priority
    src/main.cu
    src/sclp.cu
    src/graph.cpp
)
target_compile_definitions(sclp_diag_priority PRIVATE
    SCLP_MERGE_DIAGNOSTICS=1
    SCLP_PRIORITY_ORIENTATION=1
)

foreach(target multilevel_lp sclp_diag_mr sclp_diag_priority)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>
    )
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()
