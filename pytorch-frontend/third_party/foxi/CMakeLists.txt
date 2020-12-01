# Minimum CMake required
cmake_minimum_required(VERSION 3.1)
# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(
    CMAKE_BUILD_TYPE "Release"
    CACHE
      STRING
      "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage."
    FORCE)
endif()
cmake_policy(SET CMP0063 NEW)

# Project
project(foxi C)

option(FOXI_WERROR "Build with Werror" OFF)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

set(FOXI_ROOT ${PROJECT_SOURCE_DIR})

# Read ONNX version
file(READ "${PROJECT_SOURCE_DIR}/VERSION_NUMBER" ONNX_VERSION)
string(STRIP "${ONNX_VERSION}" ONNX_VERSION)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  if(ONNX_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif()
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export include directories
# ---[ ONNX Interface for Framework Integratin (FOXI)
add_library(foxi INTERFACE)
target_include_directories(foxi INTERFACE
  $<BUILD_INTERFACE:${FOXI_ROOT}>
  $<INSTALL_INTERFACE:include>)

# ---[ FOXI loader
add_library(foxi_loader STATIC foxi/onnxifi_loader.c)
target_include_directories(foxi_loader PUBLIC
  $<BUILD_INTERFACE:${FOXI_ROOT}>
  $<INSTALL_INTERFACE:include>)
# Users of ONNX backend API would compile it with their toolchain, so it is
# implemented in standard C89 for maximum compatibility
set_target_properties(foxi_loader
                      PROPERTIES C_STANDARD
                                 90
                                 C_EXTENSIONS
                                 NO)
target_link_libraries(foxi_loader PUBLIC foxi ${CMAKE_DL_LIBS})
if(MSVC)
  add_msvc_runtime_flag(foxi_loader)
endif()

if (NOT ANDROID AND NOT IOS)
  # ---[ FOXI wrapper
  add_library(foxi_wrapper MODULE foxi/onnxifi_wrapper.c)
  target_include_directories(foxi_wrapper PRIVATE
    $<BUILD_INTERFACE:${FOXI_ROOT}>
    $<INSTALL_INTERFACE:include>)
  set_target_properties(foxi_wrapper PROPERTIES
    C_STANDARD 99
    C_EXTENSIONS NO
    OUTPUT_NAME "foxi"
    POSITION_INDEPENDENT_CODE YES)
  target_link_libraries(foxi_wrapper PRIVATE foxi_loader foxi)
  if(DEFINED FOXI_SEARCH_DIR)
    target_compile_definitions(foxi_wrapper PRIVATE "FOXI_SEARCH_DIR=\"${FOXI_SEARCH_DIR}\"")
  endif()
  if(WIN32)
    if(MSVC)
      target_compile_definitions(foxi_wrapper PRIVATE "ONNXIFI_PUBLIC=__declspec(dllexport)")
    else()
      target_compile_definitions(foxi_wrapper PRIVATE "ONNXIFI_PUBLIC=__attribute__((__dllexport__))")
    endif()
  endif()
  if(APPLE)
    # By default CMake would use .so suffix on Mac
    set_target_properties(foxi_wrapper PROPERTIES SUFFIX ".dylib")
  endif()
endif()

# ---[ FOXI dummy backend
add_library(foxi_dummy SHARED foxi/onnxifi_dummy.c)
target_include_directories(foxi_dummy PRIVATE
  $<BUILD_INTERFACE:${FOXI_ROOT}>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(foxi_dummy PUBLIC foxi ${CMAKE_DL_LIBS})
target_compile_definitions(foxi_dummy PRIVATE ONNXIFI_BUILD_LIBRARY=TRUE)
if(MSVC)
  add_msvc_runtime_flag(foxi_dummy)
endif()

install(DIRECTORY ${FOXI_ROOT}/foxi
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/foxi
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h")

install(TARGETS
  foxi foxi_dummy foxi_loader
  EXPORT ONNXTargets DESTINATION lib)

if(NOT ANDROID AND NOT IOS)
  install(TARGETS foxi_wrapper
    EXPORT ONNXTargets DESTINATION lib)
endif()

set(FOXI_INCLUDE_DIRS "${FOXI_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}")
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set(FOXI_INCLUDE_DIRS ${FOXI_INCLUDE_DIRS} PARENT_SCOPE)
endif()

include(cmake/summary.cmake)
onnx_print_configuration_summary()
