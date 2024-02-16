cmake_minimum_required(VERSION 3.20)

#=========================== PROJECT ===========================#
project(STONNE
    VERSION 1.0
    DESCRIPTION "Simulation Tool for Neural Networks Engines"
    LANGUAGES CXX
)

#===================== CMAKE CONFIGURATION ====================#
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
  set(CMAKE_CXX_EXTENSIONS OFF)

  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif ()
endif ()

if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif ()

#=========================== BUILD ============================#
add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(doc)