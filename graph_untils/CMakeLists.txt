cmake_minimum_required(VERSION 3.16)
project(GraphUtils LANGUAGES CXX)

find_package(ZLIB REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(generate_graph generate_graph.cpp)
add_executable(analyze_partition analyze_partition.cpp)
add_executable(csr_to_metis csr_to_metis.cpp)
add_executable(symmetric_csr_to_metis symmetric_csr_to_metis.cpp)
add_executable(csr_symmetrize csr_symmetrize.cpp)
add_executable(edge_list_to_csr edge_list_to_csr.cpp)
target_link_libraries(edge_list_to_csr PRIVATE ZLIB::ZLIB)
