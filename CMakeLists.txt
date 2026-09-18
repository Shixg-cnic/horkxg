cmake_minimum_required(VERSION 3.18)

project(gpart LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(GPART_METIS_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../sotas/Jet-Partitioner/dependencies/install/metis"
    CACHE PATH "Existing METIS installation used by initial partitioning"
)
find_path(GPART_METIS_INCLUDE_DIR metis.h
    HINTS "${GPART_METIS_ROOT}/include"
)
find_library(GPART_METIS_LIBRARY metis
    HINTS "${GPART_METIS_ROOT}/lib"
)
find_library(GPART_GKLIB_LIBRARY GKlib
    HINTS "${GPART_METIS_ROOT}/lib"
)
if(NOT GPART_METIS_INCLUDE_DIR OR NOT GPART_METIS_LIBRARY OR
   NOT GPART_GKLIB_LIBRARY)
    message(FATAL_ERROR
        "Existing METIS was not found. Checked root: ${GPART_METIS_ROOT}; "
        "header=${GPART_METIS_INCLUDE_DIR}; library=${GPART_METIS_LIBRARY}; "
        "GKlib=${GPART_GKLIB_LIBRARY}")
endif()
message(STATUS "METIS header: ${GPART_METIS_INCLUDE_DIR}/metis.h")
message(STATUS "METIS library: ${GPART_METIS_LIBRARY}")
message(STATUS "METIS GKlib dependency: ${GPART_GKLIB_LIBRARY}")

add_library(METIS::METIS SHARED IMPORTED)
set_target_properties(METIS::METIS PROPERTIES
    IMPORTED_LOCATION "${GPART_METIS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GPART_METIS_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${GPART_GKLIB_LIBRARY}"
)

set(GPART_COARSEN_CORE_SOURCES
    src/coarsen.cu
    src/graph.cpp
)
set(GPART_COARSEN_SOURCES
    app/coarsen_main.cu
    ${GPART_COARSEN_CORE_SOURCES}
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

add_library(gpart_initial_partition STATIC
    src/initial_partition.cpp
)
target_include_directories(gpart_initial_partition PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl
)
target_link_libraries(gpart_initial_partition PUBLIC METIS::METIS)
target_compile_options(gpart_initial_partition PRIVATE -O3 -march=native)

add_executable(gpart_test_initial_partition
    tests/test_initial_partition.cpp
)
target_link_libraries(gpart_test_initial_partition PRIVATE
    gpart_initial_partition
)

# Temporary in-memory integration driver. It deliberately does not read or
# write a new standalone hierarchy format and performs no uncoarsening.
add_executable(gpart_coarsen_initial_standard
    app/coarsen_initial_main.cu
    ${GPART_COARSEN_CORE_SOURCES}
)
target_include_directories(gpart_coarsen_initial_standard PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(gpart_coarsen_initial_standard PRIVATE
    GPART_STANDARD_GRAPH
)
target_compile_options(gpart_coarsen_initial_standard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>
)
target_link_libraries(gpart_coarsen_initial_standard PRIVATE
    gpart_initial_partition
)
set_target_properties(gpart_coarsen_initial_standard PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set(GPART_PARTITION_SOURCES
    app/partition_main.cu
    ${GPART_COARSEN_CORE_SOURCES}
    src/refine.cu
    src/uncoarsen.cu
)

function(add_gpart_partition target graph_definition)
    add_executable(${target} ${GPART_PARTITION_SOURCES})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(${target} PRIVATE ${graph_definition})
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>
    )
    target_link_libraries(${target} PRIVATE gpart_initial_partition)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

add_gpart_partition(gpart_partition_standard GPART_STANDARD_GRAPH)
add_gpart_partition(gpart_partition_big GPART_BIG_GRAPH)

function(add_gpart_uncoarsen_test target graph_definition)
    add_executable(${target}
        tests/test_uncoarsen.cu
        src/coarsen.cu
        src/refine.cu
        src/uncoarsen.cu
        src/graph.cpp
    )
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

add_gpart_uncoarsen_test(
    gpart_test_uncoarsen_standard GPART_STANDARD_GRAPH)
add_gpart_uncoarsen_test(
    gpart_test_uncoarsen_big GPART_BIG_GRAPH)

enable_testing()
add_test(NAME initial_partition_small COMMAND gpart_test_initial_partition)
add_test(NAME uncoarsen_small_standard COMMAND gpart_test_uncoarsen_standard)
add_test(NAME uncoarsen_small_big COMMAND gpart_test_uncoarsen_big)

function(add_gpart_pair_partner_test target graph_definition)
    add_executable(${target} tests/test_pair_partner.cu src/graph.cpp)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(${target} PRIVATE ${graph_definition})
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native>)
    add_test(NAME ${target} COMMAND ${target})
endfunction()
add_gpart_pair_partner_test(gpart_test_pair_partner_standard GPART_STANDARD_GRAPH)
add_gpart_pair_partner_test(gpart_test_pair_partner_big GPART_BIG_GRAPH)
