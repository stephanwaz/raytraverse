cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_VERBOSE_MAKEFILE on)

project(raytraverse)


include_directories(/usr/local/include)
link_directories(/usr/local/lib)

include_directories(ray/src/rt ray/src/common)

message("${radiance_SOURCE_DIR}/ray/src/rt/Version.c")

macro(create_version_file version_file)
    add_custom_command(
            OUTPUT "${version_file}"
            COMMAND "${CMAKE_COMMAND}"
            -DRADIANCE_VERSION="${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_TWEAK}"
            -DVERSION_OUT_FILE="${version_file}"
            -DVERSION_IN_FILE="${radiance_SOURCE_DIR}/ray/src/rt/VERSION"
            -DVERSION_GOLD="${radiance_SOURCE_DIR}/ray/src/rt/Version.c"
            -P "${radiance_SOURCE_DIR}/ray/src/rt/create_version.cmake"
    )
endmacro()

set(BUILD_HEADLESS true)

add_subdirectory(ray/src/common)
add_subdirectory(ray/src/rt)



find_package(pybind11)
include_directories("${pybind11_INCLUDE_DIRS}")

pybind11_add_module(craytraverse pyhelpers.cpp)

add_library(raymod STATIC rtinit.h rtinit.c)
add_library(rcmod STATIC rcinit.h rcinit.c)

if(WIN32)
    set(rayp_SOURCES raypwin.c)
else()
    set(rayp_SOURCES raypcalls.c)
endif()

set(RCCALLFILES ${rayp_SOURCES} rayfifo.c)
list(TRANSFORM RCCALLFILES PREPEND ray/src/rt/)
add_library(rcraycalls rcraycalls.c ${RCCALLFILES})
# link in radiance libraries
#link_libraries(raycalls radiance rtrad)
# required to build rtcall
set(RTFILES duphead.c persist.c source.c pmapray.c)
set(RCFILES rc2.c)
list(TRANSFORM RTFILES PREPEND ray/src/rt/)
list(TRANSFORM RCFILES PREPEND ray/src/rt/)


add_executable(rtracemain main.cpp render.cpp rtrace.cpp rtinit.c rtraceparts.c ${RTFILES})
target_link_libraries(rtracemain PRIVATE pybind11::embed raycalls radiance rtrad raymod)

add_executable(rcontribmain mainrc.cpp render.cpp rcontrib.cpp rcinit.c rcontribparts.c rc3.c ${RCFILES})
target_link_libraries(rcontribmain PRIVATE pybind11::embed rcraycalls radiance rcmod rtrad)

pybind11_add_module(rtrace_c render.cpp rtrace.cpp rtinit.c rtraceparts.c ${RTFILES})
target_link_libraries(rtrace_c PRIVATE raycalls radiance rtrad raymod)

pybind11_add_module(rcontrib_c render.cpp rcontrib.cpp rcinit.c rcontribparts.c rc3.c ${RCFILES})
target_link_libraries(rcontrib_c PRIVATE rcraycalls rcmod radiance rtrad)

IF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    set_target_properties(craytraverse PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/")
    set_target_properties(rtrace_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/crenderer")
    set_target_properties(rcontrib_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/crenderer")
ENDIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)