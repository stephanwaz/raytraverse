function(pybind11_add_module_libs target_name)
  set(options "MODULE;SHARED;EXCLUDE_FROM_ALL;NO_EXTRAS;SYSTEM;THIN_LTO;OPT_SIZE;")
  set(oneValueArgs "SUBMODULEPATH;")
  set(multiValueArgs "LINKLIBS;")
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(ARG_MODULE AND ARG_SHARED)
    message(FATAL_ERROR "Can't be both MODULE and SHARED")
  elseif(ARG_SHARED)
    set(lib_type SHARED)
  elseif(ARG_MODULE)
    set(lib_type MODULE)
  else()
    set(lib_type STATIC)
  endif()

  if(ARG_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  else()
    set(exclude_from_all "")
  endif()

  add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})

  target_link_libraries(${target_name} PRIVATE pybind11::module)
  target_link_libraries(${target_name} PRIVATE ${ARG_LINKLIBS})

  if(ARG_SYSTEM)
    message(
            STATUS
            "Warning: this does not have an effect - use NO_SYSTEM_FROM_IMPORTED if using imported targets"
    )
  endif()

  pybind11_extension(${target_name})

  # -fvisibility=hidden is required to allow multiple modules compiled against
  # different pybind versions to work properly, and for some features (e.g.
  # py::module_local).  We force it on everything inside the `pybind11`
  # namespace; also turning it on for a pybind module compilation here avoids
  # potential warnings or issues from having mixed hidden/non-hidden types.
  set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET "hidden"
          CUDA_VISIBILITY_PRESET "hidden")

  if(ARG_NO_EXTRAS)
    return()
  endif()

  if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    if(ARG_THIN_LTO)
      target_link_libraries(${target_name} PRIVATE pybind11::thin_lto)
    else()
      target_link_libraries(${target_name} PRIVATE pybind11::lto)
    endif()
  endif()

  if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    pybind11_strip(${target_name})
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

  if(ARG_OPT_SIZE)
    target_link_libraries(${target_name} PRIVATE pybind11::opt_size)
  endif()
endfunction()
