cmake_minimum_required(VERSION 3.18)

project(gpart LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(GPART_COARSEN_SOURCES
    app/coarsen_main.cu
    src/coarsen.cu
    src/graph.cpp
)

# graph.cpp owns host/device graph transfer helpers and therefore needs the
# CUDA/Thrust include environment even though it contains no kernels.
set_source_files_properties(src/graph.cpp PROPERTIES LANGUAGE CUDA)

function(add_gpart_coarsen target graph_definition)
    add_executable(${target} ${GPART_COARSEN_SOURCES})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(${target} PRIVATE ${graph_definition})
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>
    )
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

add_gpart_coarsen(gpart_coarsen_standard GPART_STANDARD_GRAPH)
add_gpart_coarsen(gpart_coarsen_big GPART_BIG_GRAPH)
