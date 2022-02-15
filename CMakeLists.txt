cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
project(raytraverse)

add_compile_definitions(VERSION_INFO="1.2.5")

add_subdirectory(src)
set(CMAKE_VERBOSE_MAKEFILE on)

install(TARGETS rtrace_c rcontrib_c LIBRARY DESTINATION raytraverse/crenderer)
install(TARGETS craytraverse LIBRARY DESTINATION raytraverse)
install(TARGETS rtrace rcontrib  total cnt rcalc getinfo vwrays
    pvalue pcompos pcomb pfilt oconv gendaylit xform RUNTIME DESTINATION bin)

