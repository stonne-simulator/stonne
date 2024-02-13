cmake_minimum_required(VERSION 3.20)

#=========================== PROJECT ===========================#
project(STONNE LANGUAGES CXX)

#==================== COMPILER CONFIGURATION ===================#
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

#===================== CMAKE CONFIGURATION ====================#
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

#========================== LIBRARIES =========================#

#====================== PROJECT FOLDERS =======================#
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")
set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")

# TODO: remove later
include_directories(${INCLUDE_DIR})
include_directories(${SOURCE_DIR}/stonne)

#===================== BUILD EXECUTABLES ======================#
add_subdirectory(src)
#add_subdirectory(tests)