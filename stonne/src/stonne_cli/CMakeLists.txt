cmake_minimum_required(VERSION 3.20)
set(EXECUTABLE "stonne_cli")

#====================== SETUP EXECUTABLE ======================#
add_executable(${EXECUTABLE})
target_compile_features(${EXECUTABLE} PRIVATE cxx_std_20)
# TODO: uncomment this later
#target_compile_options(${EXECUTABLE} PRIVATE -Wall -Wextra -Wpedantic -Werror)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(${EXECUTABLE} PRIVATE -g -fno-omit-frame-pointer)
else (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(${EXECUTABLE} PRIVATE -O3 -march=native -mtune=native)
endif ()

#=================== COMPILER CONFIGURATION =====================#
set_target_properties(${EXECUTABLE}
    PROPERTIES
    CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE
    CMAKE_CXX_EXTENSIONS OFF
)

#======================== INCLUDE DIRS ==========================#
target_include_directories(${EXECUTABLE}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

#======================= LINK LIBRARIES ========================#
target_link_libraries(${EXECUTABLE} PRIVATE stonne)

#======================== SOURCE FILES =========================#
target_sources(${EXECUTABLE} PRIVATE
    stonne_cli.cpp
)