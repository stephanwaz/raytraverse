cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(raytraverse)

set(PYBIND11_PYTHON_VERSION 3.6)

find_package(X11)
if(X11_FOUND)
    if(APPLE)
        message(${X11_INCLUDE_DIR})
        include_directories(${X11_INCLUDE_DIR})
        endif()
    endif()

# for mac os
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(MACOSX_DEPLOYMENT_TARGET 10.9)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
endif()

add_subdirectory(pybind11)
include_directories(pybind11/include/pybind11)
include(pybind11_add_module_libs.cmake)

include_directories(Radiance/src/common Radiance/src/util)

execute_process(COMMAND git log --pretty=format:'%h' -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Radiance
        OUTPUT_VARIABLE RADIANCE_VERSION
        ERROR_QUIET)

macro(create_version_file version_file)
    add_custom_command(
            OUTPUT "${version_file}"
            COMMAND "${CMAKE_COMMAND}"
            -DRADIANCE_VERSION="${RADIANCE_VERSION}"
            -DVERSION_OUT_FILE="${version_file}"
            -DVERSION_IN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/Radiance/src/rt/VERSION"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/create_version.cmake"
    )
endmacro()


set(VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/Version.c")
create_version_file("${VERSION_FILE}")

# make lists of radiance source files for library creation
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

set(RADIANCE_COMMON paths.c platform.h random.h)
set(RADIANCE_RT
        ambcomp.c ambient.c ambio.c aniso.c ashikhmin.c data.c dielectric.c fprism.c
        freeobjmem.c func.c glass.c initotypes.c m_alias.c m_brdf.c m_bsdf.c m_clip.c
        m_direct.c m_mirror.c m_mist.c mx_data.c mx_func.c noise3.c normal.c o_cone.c o_face.c
        o_instance.c o_mesh.c p_data.c p_func.c pmap.c pmapamb.c pmapbias.c pmapcontrib.c pmapdata.c
        pmapdiag.c pmapio.c pmapmat.c pmapopt.c pmapparm.c pmaprand.c pmapray.c pmapsrc.c pmaptype.c
        pmcontrib2.c pmutil.c preload.c renderopts.c source.c sphere.c srcobstr.c srcsamp.c
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
#list(APPEND rtrad_SOURCES csrc/calcompcal.c)
#list(APPEND RADIANCE_RT csrc/func_mod.c)
list(APPEND RADIANCE_RT csrc/raytrace_lr0.c)

add_library(rtrad STATIC ${rtrad_SOURCES})
add_library(radiance STATIC ${VERSION_FILE} ${RADIANCE_COMMON} ${RADIANCE_RT})
add_library(raycalls STATIC ${RAYCALLS})
add_library(rcraycalls STATIC csrc/rcraycalls.c ${RCCALLFILES})
add_library(rtracecfiles STATIC csrc/rtinit.c csrc/rtraceparts.c ${RTFILES} ${VERSION_FILE})
add_library(rcontribcfiles STATIC csrc/rcinit.c csrc/rcontribparts.c csrc/rc3_mod.c csrc/rc2_mod.c ${VERSION_FILE})
target_include_directories(rtrad PRIVATE Radiance/src/rt)
target_include_directories(radiance PRIVATE Radiance/src/rt)
target_include_directories(raycalls PRIVATE Radiance/src/rt)
target_include_directories(rcraycalls PRIVATE Radiance/src/rt)
target_include_directories(rtracecfiles PRIVATE Radiance/src/rt)
target_include_directories(rcontribcfiles PRIVATE Radiance/src/rt)

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
endif()

# target for building 3 craytraverse modules
ADD_CUSTOM_TARGET(python_all)
ADD_DEPENDENCIES(python_all rtrace_c rcontrib_c craytraverse)

# build required radiance executables

# from rt
set(RTRACEFILES rtmain.c rtrace.c duphead.c persist.c)
list(TRANSFORM RTRACEFILES PREPEND Radiance/src/rt/)
add_executable(rtrace ${RTRACEFILES})
target_link_libraries(rtrace raycalls radiance rtrad)

set(RCONTRIBFILES rcmain.c rcontrib.c rc2.c rc3.c)
list(TRANSFORM RCONTRIBFILES PREPEND Radiance/src/rt/)
add_executable(rcontrib ${RCONTRIBFILES})
target_link_libraries(rcontrib radiance rtrad)

# from util
add_executable(getinfo Radiance/src/util/getinfo.c)
target_link_libraries(getinfo rtrad)

add_executable(vwrays Radiance/src/util/vwrays.c)
target_link_libraries(vwrays rtrad)


set(libreetz_SOURCES g3affine.c g3affine.h g3flist.c g3flist.h g3list.h g3nlist.h
        g3sphere.c g3sphere.h g3vector.c g3vector.h gbasic.c gbasic.h muc_randvar.c muc_randvar.h)


list(TRANSFORM libreetz_SOURCES PREPEND Radiance/src/common/)

add_library(libreetz STATIC ${libreetz_SOURCES})

add_executable(evalglare csrc/evalglare.c Radiance/src/util/pictool.c)
target_link_libraries(evalglare rtrad libreetz)

# from cal
add_executable(cnt Radiance/src/cal/cnt.c)

add_executable(total Radiance/src/cal/total.c)
target_link_libraries(total rtrad ${LIB_M})

add_executable(rcalc Radiance/src/cal/rcalc.c)
target_link_libraries(rcalc rtrad)

# from px
add_executable(pvalue Radiance/src/px/pvalue.c)
target_link_libraries(pvalue rtrad)

add_executable(pcompos Radiance/src/px/pcompos.c)
target_link_libraries(pcompos rtrad)

add_executable(pcomb Radiance/src/px/pcomb.c)
target_link_libraries(pcomb rtrad)

set(PFILTFILES pf2.c pf3.c pfilt.c)
list(TRANSFORM PFILTFILES PREPEND Radiance/src/px/)
add_executable(pfilt ${PFILTFILES})
target_link_libraries(pfilt rtrad)

# from ot
set(OCONVFILES bbox.c initotypes.c o_cone.c o_face.c o_instance.c oconv.c sphere.c writeoct.c)
list(TRANSFORM OCONVFILES PREPEND Radiance/src/ot/)
add_executable(oconv ${OCONVFILES})
target_link_libraries(oconv rtrad)

# from gen
add_executable(gendaylit Radiance/src/gen/gendaylit.c Radiance/src/gen/sun.c)
target_link_libraries(gendaylit rtrad ${LIB_M})

add_executable(xform Radiance/src/gen/xform.c)
target_link_libraries(xform rtrad)

#if(NOT SKBUILD)
#
#set(ALLTARGETS rtrad radiance raycalls rcraycalls rtracecfiles rcontribcfiles rtrace_c rcontrib_c
#        craytraverse rtrace rcontrib  total cnt rcalc getinfo vwrays pvalue pcompos pcomb oconv gendaylit)
#
#set(ALLSOURCES)
#foreach(TGT IN ITEMS ${ALLTARGETS})
#    get_target_property(TGTSOURCES ${TGT} SOURCES)
#    foreach(SRC IN ITEMS ${TGTSOURCES})
#        list(APPEND ALLSOURCES ${SRC})
#        endforeach()
#    endforeach()
#list(REMOVE_DUPLICATES ALLSOURCES)
#list(FILTER ALLSOURCES EXCLUDE REGEX ^/.*)
#list(TRANSFORM ALLSOURCES PREPEND src/)
#
#foreach(SRC IN ITEMS ${ALLSOURCES})
#    message(${SRC})
#endforeach()
#
#string(REPLACE ";" "\n" TOWRITE "${ALLSOURCES}")
#file(WRITE MANIFEST "${TOWRITE}")
#
#endif()
