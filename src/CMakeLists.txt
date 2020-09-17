cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE on)
project(raytraverse)

# modify these based on your system
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

include_directories(Radiance/src/rt Radiance/src/common)

macro(create_version_file version_file)
    add_custom_command(
            OUTPUT "${version_file}"
            COMMAND "${CMAKE_COMMAND}"
            -DRADIANCE_VERSION="${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_TWEAK}"
            -DVERSION_OUT_FILE="${version_file}"
            -DVERSION_IN_FILE="${radiance_SOURCE_DIR}/Radiance/src/rt/VERSION"
            -DVERSION_GOLD="${radiance_SOURCE_DIR}/Radiance/src/rt/Version.c"
            -P "${radiance_SOURCE_DIR}/Radiance/src/rt/create_version.cmake"
    )
endmacro()

# used by setup.py for distribution
if(NOT SKBUILD)
    set(STATIC_VERSION_FILE "../StaticVersion.c")
    create_version_file("${STATIC_VERSION_FILE}")
else()
    set(STATIC_VERSION_FILE "StaticVersion.c")
endif(NOT SKBUILD)


set(BUILD_HEADLESS true)

# create libraries needed for compilation from radiance source

set(rtrad_SOURCES
        addobjnotify.c badarg.c biggerlib.c bmalloc.c bmpfile.c bsdf.c bsdf_m.c
        bsdf_t.c byteswap.c caldefn.c calexpr.c calfunc.c calprnt.c ccolor.c ccyrgb.c
        chanvalue.c clip.c color.c colrops.c cone.c cvtcmd.c depthcodec.c dircode.c disk2square.c
        ealloc.c eputs.c erf.c error.c expandarg.c ezxml.c face.c falsecolor.c fdate.c fgetline.c
        fgetval.c fgetword.c fixargv0.c fltdepth.c font.c fputword.c free_os.c fropen.c fvect.c
        gethomedir.c getlibpath.c getpath.c header.c hilbert.c idmap.c image.c instance.c interp2d.c
        invmat4.c lamps.c linregr.c loadbsdf.c loadvars.c lookup.c mat4.c mesh.c modobject.c multisamp.c
        myhostname.c normcodec.c objset.c octree.c otypes.c paths.c plocate.c portio.c process.c
        quit.c readfargs.c readmesh.c readobj.c readoct.c resolu.c rexpr.c savestr.c savqstr.c
        sceneio.c spec_rgb.c tcos.c timegm.c tmap16bit.c tmapcolrs.c tmapluv.c tmaptiff.c tmesh.c
        tonemap.c triangulate.c urand.c urind.c wordfile.c words.c wputs.c xf.c zeroes.c
        )
if(UNIX)
    find_library(LIB_M m DOC "Path to libm")
    if(NOT LIB_M)
        message(FATAL_ERROR "Cannot build radiance without libm.  Please set LIB_M")
    endif()
    list(APPEND rtrad_SOURCES unix_process.c)
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        list(APPEND rtrad_SOURCES strlcpy.c)
    endif()
else()
    list(APPEND rtrad_SOURCES win_process.c win_popen.c win_usleep.c strlcpy.c)
    set(LIB_M)
endif()

list(TRANSFORM rtrad_SOURCES PREPEND Radiance/src/common/)
add_library(rtrad ${rtrad_SOURCES})
target_link_libraries(rtrad ${LIB_M})


set(RADIANCE_COMMON paths.c platform.h random.h)
set(RADIANCE_RT
        ambcomp.c ambient.c ambio.c aniso.c ashikhmin.c data.c dielectric.c fprism.c
        freeobjmem.c func.c glass.c initotypes.c m_alias.c m_brdf.c m_bsdf.c m_clip.c
        m_direct.c m_mirror.c m_mist.c mx_data.c mx_func.c noise3.c normal.c o_cone.c o_face.c
        o_instance.c o_mesh.c p_data.c p_func.c pmap.c pmapamb.c pmapbias.c pmapcontrib.c pmapdata.c
        pmapdiag.c pmapio.c pmapmat.c pmapopt.c pmapparm.c pmaprand.c pmapray.c pmapsrc.c pmaptype.c
        pmcontrib2.c pmutil.c preload.c raytrace.c renderopts.c source.c sphere.c srcobstr.c srcsamp.c
        srcsupp.c t_data.c t_func.c text.c virtuals.c
        )
list(TRANSFORM RADIANCE_RT PREPEND Radiance/src/rt/)
list(TRANSFORM RADIANCE_COMMON PREPEND Radiance/src/common/)
add_library(radiance ${STATIC_VERSION_FILE} ${RADIANCE_COMMON} ${RADIANCE_RT})

if(WIN32)
    set(rayp_SOURCES raypwin.c)
else()
    set(rayp_SOURCES raypcalls.c)
endif()

set(RAYCALLS raycalls.c ${rayp_SOURCES} rayfifo.c)
list(TRANSFORM RAYCALLS PREPEND Radiance/src/rt/)
add_library(raycalls ${RAYCALLS})


# if pybind is not installed on system, then clone and add_subdirectory instead
add_subdirectory(pybind11)
include_directories(pybind11/include/pybind11)

pybind11_add_module(craytraverse pyhelpers.cpp)


if(WIN32)
    set(rayp_SOURCES raypwin.c)
else()
    set(rayp_SOURCES raypcalls.c)
endif()

# rcontrib needs a seperate call library from rtrace to avoid redundant declerations
# in the radiance source.
set(RCCALLFILES ${rayp_SOURCES} rayfifo.c)
list(TRANSFORM RCCALLFILES PREPEND Radiance/src/rt/)
add_library(rcraycalls rcraycalls.c ${RCCALLFILES})

set(RTFILES duphead.c persist.c source.c pmapray.c)
list(TRANSFORM RTFILES PREPEND Radiance/src/rt/)
list(TRANSFORM RCFILES PREPEND Radiance/src/rt/)

# rtrace and rcontrib are built seperately to avoid namespace conflicts
add_library(rtracecfiles rtinit.c rtraceparts.c ${RTFILES} ${STATIC_VERSION_FILE})
pybind11_add_module(rtrace_c render.cpp rtrace.cpp)
target_link_libraries(rtrace_c PRIVATE raycalls radiance rtrad rtracecfiles)

add_library(rcontribcfiles rcinit.c rcontribparts.c rc3.c rc2.c ${STATIC_VERSION_FILE})
pybind11_add_module(rcontrib_c render.cpp rcontrib.cpp)
target_link_libraries(rcontrib_c PRIVATE rcraycalls radiance rtrad rcontribcfiles)

# target for building 3 craytraverse modules
ADD_CUSTOM_TARGET(python_all)
ADD_DEPENDENCIES(python_all rtrace_c rcontrib_c craytraverse)

# move submodules to raytraverse
IF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    set_target_properties(craytraverse PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/")
    set_target_properties(rtrace_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/crenderer")
    set_target_properties(rcontrib_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/crenderer")
ENDIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)

# these are for testing buy building as an exectuble, should function the same as rtrace/rcontrib
# except input is given as last argument instead of stdin
if(NOT SKBUILD)
    add_executable(rtracemain main.cpp render.cpp rtrace.cpp)
    target_link_libraries(rtracemain PRIVATE pybind11::embed raycalls radiance rtrad rtracecfiles)
    add_executable(rcontribmain mainrc.cpp render.cpp rcontrib.cpp)
    target_link_libraries(rcontribmain PRIVATE pybind11::embed rcraycalls radiance rtrad rcontribcfiles)
endif(NOT SKBUILD)
