cmake_minimum_required(VERSION 3.20)
set(LIBSTONNE "libstonne")

#======================== SETUP LIBRARY ========================#
add_library(${LIBSTONNE} STATIC)
target_compile_features(${LIBSTONNE} PRIVATE cxx_std_20)
target_compile_options(${LIBSTONNE} PRIVATE -fopenmp)
# TODO: uncomment later
#target_compile_options(${LIBSTONNE} PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(${LIBSTONNE} PRIVATE -Wno-unused-parameter)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(${LIBSTONNE} PRIVATE -g -fno-omit-frame-pointer)
else (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(${LIBSTONNE} PRIVATE -O3 -ffast-math -march=native -mtune=native)
  add_definitions(-D NDEBUG) # Disable asserts
endif ()

#======================== INCLUDE DIRS ==========================#
target_include_directories(${LIBSTONNE}
    PRIVATE
    ${INCLUDE_DIR}/stonne
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EXTERNAL_DIR}
)

#======================== SOURCE FILES ==========================#
target_sources(${LIBSTONNE} PRIVATE
    ASNetwork.cpp
    ASwitch.cpp
    AccumulationBuffer.cpp
    Accumulator.cpp
    CollectionBus.cpp
    CollectionBusLine.cpp
    CompilerART.cpp
    CompilerFEN.cpp
    CompilerMSN.cpp
    CompilerMultiplierMesh.cpp
    Config.cpp
    Connection.cpp
    DNNLayer.cpp
    DNNModel.cpp
    DSNetwork.cpp
    DSNetworkTop.cpp
    DSwitch.cpp
    DataPackage.cpp
    FEASwitch.cpp
    FENetwork.cpp
    Fifo.cpp
    GustavsonsSpGEMMSDMemory.cpp
    LookupTable.cpp
    MSNetwork.cpp
    MSwitch.cpp
    MultiplierOS.cpp
    OSMeshMN.cpp
    OSMeshSDMemory.cpp
    OuterLoopSpGEMMSDMemory.cpp
    SDMemory.cpp
    STONNEModel.cpp
    SparseDenseSDMemory.cpp
    SparseFlex_ASNetwork.cpp
    SparseFlex_ASwitch.cpp
    SparseFlex_MSNetwork.cpp
    SparseFlex_MSwitch.cpp
    SparseSDMemory.cpp
    Stats.cpp
    TemporalRN.cpp
    Tile.cpp
    testbench.cpp
    utility.cpp
    TileGenerator/StonneMapper/StonneMapperGenerator.cpp
    TileGenerator/StonneMapper/TileOptionDenseGemm.cpp
    TileGenerator/StonneMapper/TileOptionSparseDense.cpp
    TileGenerator/StonneMapper/utility.cpp
    TileGenerator/TileGenerator.cpp
    TileGenerator/mRNA/Analyzer.cpp
    TileGenerator/mRNA/DNNModel.cpp
    TileGenerator/mRNA/DSNetwork.cpp
    TileGenerator/mRNA/MAERIModel.cpp
    TileGenerator/mRNA/MSNetwork.cpp
    TileGenerator/mRNA/MappingStrategy.cpp
    TileGenerator/mRNA/MrnaGenerator.cpp
    TileGenerator/mRNA/RSNetwork.cpp
    TileGenerator/mRNA/SDMemory.cpp
    TileGenerator/mRNA/VNAT.cpp
)