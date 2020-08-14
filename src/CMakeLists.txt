cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_VERBOSE_MAKEFILE on)

project(raytraverse)


include_directories(/usr/local/include)
link_directories(/usr/local/lib)

include_directories(ray/src/rt ray/src/common)

add_subdirectory(ray/src/common)
add_subdirectory(ray/src/rt)



find_package(pybind11)
include_directories("${pybind11_INCLUDE_DIRS}")

# link in radiance libraries
link_libraries(raycalls radiance rtrad)
# required to build rtcall
set(RTFILES rtrace.c duphead.c persist.c)
list(TRANSFORM RTFILES PREPEND ray/src/rt/)

pybind11_add_module(craytraverse craytraverse.cpp pyhelpers.cpp radiance.cpp render.cpp rtrace.cpp rtcall.c ${RTFILES})


IF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    set_target_properties(craytraverse PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/")
ENDIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)