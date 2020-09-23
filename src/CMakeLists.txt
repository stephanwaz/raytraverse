cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(raytraverse)

set(PYBIND11_PYTHON_VERSION 3.6)

# for mac os
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(MACOSX_DEPLOYMENT_TARGET 10.9)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
endif()

add_subdirectory(pybind11)
include_directories(pybind11/include/pybind11)
include(pybind11_add_module_libs.cmake)


include_directories(Radiance/src/rt Radiance/src/common)

macro(create_version_file version_file)
    add_custom_command(
            OUTPUT "${version_file}"
            COMMAND "${CMAKE_COMMAND}"
            -DRADIANCE_VERSION="${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_TWEAK}"
            -DVERSION_OUT_FILE="${version_file}"
            -DVERSION_IN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/Radiance/src/rt/VERSION"
            -DVERSION_GOLD="${CMAKE_CURRENT_SOURCE_DIR}/Radiance/src/rt/Version.c"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/Radiance/src/rt/create_version.cmake"
    )
endmacro()


set(STATIC_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/StaticVersion.c")
create_version_file("${STATIC_VERSION_FILE}")

# used by setup.py for distribution
#if(NOT SKBUILD)
#    set(STATIC_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/StaticVersion.c")
#    create_version_file("${STATIC_VERSION_FILE}")
#else()
#    set(STATIC_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/StaticVersion.c")
#endif(NOT SKBUILD)

# make lists of radiance source files for library creation
set(rtrad_SOURCES
        addobjnotify.c badarg.c biggerlib.c bmalloc.c bmpfile.c bsdf.c bsdf_m.c
        bsdf_t.c byteswap.c calexpr.c calfunc.c calprnt.c ccolor.c ccyrgb.c
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

set(RADIANCE_COMMON paths.c platform.h random.h)
set(RADIANCE_RT
        ambcomp.c ambient.c ambio.c aniso.c ashikhmin.c data.c dielectric.c fprism.c
        freeobjmem.c glass.c initotypes.c m_alias.c m_brdf.c m_bsdf.c m_clip.c
        m_direct.c m_mirror.c m_mist.c mx_data.c mx_func.c noise3.c normal.c o_cone.c o_face.c
        o_instance.c o_mesh.c p_data.c p_func.c pmap.c pmapamb.c pmapbias.c pmapcontrib.c pmapdata.c
        pmapdiag.c pmapio.c pmapmat.c pmapopt.c pmapparm.c pmaprand.c pmapray.c pmapsrc.c pmaptype.c
        pmcontrib2.c pmutil.c preload.c raytrace.c renderopts.c source.c sphere.c srcobstr.c srcsamp.c
        srcsupp.c t_data.c t_func.c text.c virtuals.c
        )

if(WIN32)
    set(rayp_SOURCES raypwin.c)
else()
    set(rayp_SOURCES raypcalls.c)
endif()

set(RAYCALLS raycalls.c ${rayp_SOURCES} rayfifo.c)
# rcontrib needs a seperate call library from rtrace to avoid redundant declerations
# in the radiance source.
set(RCCALLFILES ${rayp_SOURCES} rayfifo.c)

set(RTFILES duphead.c persist.c source.c pmapray.c)

#prepend source directories
list(TRANSFORM rtrad_SOURCES PREPEND Radiance/src/common/)
list(TRANSFORM RADIANCE_RT PREPEND Radiance/src/rt/)
list(TRANSFORM RADIANCE_COMMON PREPEND Radiance/src/common/)
list(TRANSFORM RAYCALLS PREPEND Radiance/src/rt/)
list(TRANSFORM RCCALLFILES PREPEND Radiance/src/rt/)
list(TRANSFORM RTFILES PREPEND Radiance/src/rt/)
#add additional modified sources from parent directory
list(APPEND rtrad_SOURCES calcompcal.c)
list(APPEND RADIANCE_RT func.c)


add_library(rtrad STATIC ${rtrad_SOURCES})
add_library(radiance STATIC ${STATIC_VERSION_FILE} ${RADIANCE_COMMON} ${RADIANCE_RT})
add_library(raycalls STATIC ${RAYCALLS})
add_library(rcraycalls STATIC rcraycalls.c ${RCCALLFILES})
add_library(rtracecfiles STATIC rtinit.c rtraceparts.c ${RTFILES} ${STATIC_VERSION_FILE})
add_library(rcontribcfiles STATIC rcinit.c rcontribparts.c rc3.c rc2.c ${STATIC_VERSION_FILE})


target_link_libraries(rtrad ${LIB_M})

# rtrace and rcontrib are built seperately to avoid namespace conflicts
pybind11_add_module(craytraverse pyhelpers.cpp)
pybind11_add_module_libs(rtrace_c MODULE render.cpp rtrace.cpp LINKLIBS raycalls radiance rtrad rtracecfiles)
pybind11_add_module_libs(rcontrib_c MODULE render.cpp rcontrib.cpp LINKLIBS rcraycalls radiance rtrad rcontribcfiles)





if(SKBUILD)
    set_target_properties(craytraverse PROPERTIES LIBRARY_OUTPUT_DIRECTORY "raytraverse/")
    set_target_properties(rtrace_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "raytraverse/crenderer")
    set_target_properties(rcontrib_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "raytraverse/crenderer")
else()
    # move submodules to raytraverse
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    set_target_properties(craytraverse PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/")
    set_target_properties(rtrace_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/crenderer/")
    set_target_properties(rcontrib_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PARENT_DIR}/raytraverse/crenderer/")
    # these are for testing buy building as an exectuble, should function the same as rtrace/rcontrib
    # except input is given as last argument instead of stdin
    add_executable(rtracemain main.cpp render.cpp rtrace.cpp)
    target_link_libraries(rtracemain PRIVATE pybind11::embed raycalls radiance rtrad rtracecfiles)
    add_executable(rcontribmain mainrc.cpp render.cpp rcontrib.cpp)
    target_link_libraries(rcontribmain PRIVATE pybind11::embed rcraycalls radiance rtrad rcontribcfiles)
endif()

# target for building 3 craytraverse modules
ADD_CUSTOM_TARGET(python_all)
ADD_DEPENDENCIES(python_all rtrace_c rcontrib_c craytraverse)


# Get all propreties that cmake supports
execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)

# Convert command output into a CMake list
STRING(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
STRING(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

function(print_properties)
    message ("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction(print_properties)

function(print_target_properties tgt)
    if(NOT TARGET ${tgt})
        message("There is no target named '${tgt}'")
        return()
    endif()

    foreach (prop ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})
        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(prop STREQUAL "LOCATION" OR prop MATCHES "^LOCATION_" OR prop MATCHES "_LOCATION$")
            continue()
        endif()
        # message ("Checking ${prop}")
        get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
        if (propval)
            get_target_property(propval ${tgt} ${prop})
            message ("${tgt} ${prop} = ${propval}")
        endif()
    endforeach(prop)
endfunction(print_target_properties)

print_target_properties(rcontrib_c)
print_target_properties(craytraverse)