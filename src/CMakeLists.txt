cmake_minimum_required(VERSION 3.15)

project(src)
set(CMAKE_CXX_STANDARD 14)

#get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
#
#include_directories(/usr/local/include)
#link_directories(/usr/local/lib)
#
#find_package(pybind11)
#
#include_directories("${pybind11_INCLUDE_DIRS}")
#
#set(LIBRARY_OUTPUT_DIRECTORY "/Users/stephenwasilewski/dev/raytraverse/")
#
#pybind11_add_module(craytraverse craytraverse.cpp)
#
#message("${CMAKE_BUILD_TYPE}")
#
#IF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
#    set_target_properties(craytraverse PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/")
#ENDIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c11 -std=c++11 -stdlib=libc++")
#add_definitions(-DBSD -DNOSTEREO -Dfreebsd)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.7")
#
#include_directories(/Users/stephenwasilewski/dev/compile_radiance/ray/src/rt)
#include_directories(/Users/stephenwasilewski/dev/compile_radiance/ray/src/common)
#
#

if(UNIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/radiance")
    add_definitions(-Dlinux -D_FILE_OFFSET_BITS=64 -DNOSTEREO)
else()
    set(CPACK_PACKAGING_INSTALL_PREFIX "/")
endif()

set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_TWEAK}-${CMAKE_SYSTEM_NAME}")

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c11 -std=c++11 -stdlib=libc++")
    add_definitions(-DBSD -DNOSTEREO -Dfreebsd)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.7")
endif()

if(APPLE)
    set(CPACK_GENERATOR "PackageMaker")
elseif(UNIX)
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)

add_subdirectory("ray/src")
