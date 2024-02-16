cmake_minimum_required(VERSION 3.20)
set(STONNE "stonne")

#======================== SETUP LIBRARY ========================#
add_library(${STONNE} STATIC)
target_compile_features(${STONNE} PUBLIC cxx_std_20)
# TODO: uncomment later
#target_compile_options(${STONNE} PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(${STONNE} PRIVATE -Wno-unused-parameter)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(${STONNE} PRIVATE -g -fno-omit-frame-pointer)
else (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(${STONNE} PRIVATE -O3 -march=native -mtune=native)
  add_definitions(-D NDEBUG) # Disable asserts
endif ()

#=================== COMPILER CONFIGURATION =====================#
set_target_properties(${STONNE}
    PROPERTIES
    CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE
    CMAKE_CXX_EXTENSIONS OFF
)

#======================== INCLUDE DIRS ==========================#
target_include_directories(${STONNE}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../external

    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

#======================== SOURCE FILES ==========================#
target_sources(${STONNE} PRIVATE
    network/rn/ASNetwork.cpp
    network/rn/switch/ASwitch.cpp
    memory/AccumulationBuffer.cpp
    memory/Accumulator.cpp
    comm/CollectionBus.cpp
    comm/CollectionBusLine.cpp
    compiler/CompilerART.cpp
    compiler/CompilerFEN.cpp
    compiler/CompilerMSN.cpp
    compiler/CompilerMultiplierMesh.cpp
    common/Config.cpp
    comm/Connection.cpp
    common/dnn/DNNLayer.cpp
    common/dnn/DNNModel.cpp
    network/dn/DSNetwork.cpp
    network/dn/DSNetworkTop.cpp
    network/dn/switch/DSwitch.cpp
    comm/DataPackage.cpp
    network/rn/switch/FEASwitch.cpp
    network/rn/FENetwork.cpp
    common/Fifo.cpp
    memctrl/GustavsonsSpGEMMSDMemory.cpp
    memory/LookupTable.cpp
    network/mn/MSNetwork.cpp
    network/mn/switch/MSwitch.cpp
    network/mn/MultiplierOS.cpp
    network/mn/OSMeshMN.cpp
    memctrl/OSMeshSDMemory.cpp
    memctrl/OuterLoopSpGEMMSDMemory.cpp
    memctrl/SDMemory.cpp
    STONNEModel.cpp
    memctrl/SparseDenseSDMemory.cpp
    network/rn/SparseFlex_ASNetwork.cpp
    network/rn/switch/SparseFlex_ASwitch.cpp
    network/mn/SparseFlex_MSNetwork.cpp
    network/mn/switch/SparseFlex_MSwitch.cpp
    memctrl/SparseSDMemory.cpp
    common/Stats.cpp
    network/rn/TemporalRN.cpp
    common/dnn/Tile.cpp
    common/utility.cpp
    tile_generator/stonne_mapper/StonneMapperGenerator.cpp
    tile_generator/stonne_mapper/TileOptionDenseGemm.cpp
    tile_generator/stonne_mapper/TileOptionSparseDense.cpp
    tile_generator/stonne_mapper/utility.cpp
    tile_generator/TileGenerator.cpp
    tile_generator/mRNA/Analyzer.cpp
    tile_generator/mRNA/DNNModel.cpp
    tile_generator/mRNA/DSNetwork.cpp
    tile_generator/mRNA/MAERIModel.cpp
    tile_generator/mRNA/MSNetwork.cpp
    tile_generator/mRNA/MappingStrategy.cpp
    tile_generator/mRNA/MrnaGenerator.cpp
    tile_generator/mRNA/RSNetwork.cpp
    tile_generator/mRNA/SDMemory.cpp
    tile_generator/mRNA/VNAT.cpp
)