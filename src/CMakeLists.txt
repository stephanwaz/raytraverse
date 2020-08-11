cmake_minimum_required(VERSION 3.15)
project(src)
set(CMAKE_CXX_STANDARD 14)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

find_package(pybind11)

include_directories("${pybind11_INCLUDE_DIRS}")

set(LIBRARY_OUTPUT_DIRECTORY "/Users/stephenwasilewski/dev/raytraverse/")

pybind11_add_module(craytraverse craytraverse.cpp)

IF(CMAKE_BUILD_TYPE MATCHES Release)
    set_target_properties(craytraverse PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/")
ENDIF(CMAKE_BUILD_TYPE MATCHES Release)
