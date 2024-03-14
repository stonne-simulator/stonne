
#ifndef UTILITY_H_
#define UTILITY_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "common/types.hpp"
#include "tile_generator/types.hpp"

// Parse functions
bool isNum(std::string str);

bool ispowerof2(std::size_t x);

std::size_t nextPowerOf2(int x);

std::string getstr(std::istringstream& instr);

std::string get_string_layer_t(Layer_t kernelOperation);

Layer_t get_type_layer_t(std::string kernelOperation);

std::string to_lower(std::string str);

TileGenerator::Target parseTileGeneratorTarget(std::string target);

std::string parseTileGeneratorTarget(TileGenerator::Target target);

TileGenerator::Generator parseTileGenerator(std::string generator);

std::string parseTileGenerator(TileGenerator::Generator generator);

std::string get_string_adder_configuration(adderconfig_t config);

std::string get_string_fwlink_direction(fl_t fl_direction);

std::string ind(std::size_t indent);  //Get a string with as many spaces as indent value.

std::string get_string_reduce_network_type(ReduceNetwork_t reduce_network_type);

ReduceNetwork_t get_type_reduce_network_type(std::string reduce_network_type);

MemoryController_t get_type_memory_controller_type(std::string memory_controller_type);

std::string get_string_memory_controller_type(MemoryController_t memory_controller_type);

std::string get_string_multiplier_network_type(MultiplierNetwork_t multiplier_network_type);

MultiplierNetwork_t get_type_multiplier_network_type(std::string multiplier_network_type);

Dataflow get_type_dataflow_type(std::string dataflow_type);

std::string get_string_dataflow_type(Dataflow dataflow);

float* generateMatrixDense(std::size_t rows, std::size_t cols, std::size_t sparsity);

std::size_t* generateBitMapFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type);

float* generateMatrixSparseFromDenseNoBitmap(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type, std::size_t& size);

float* generateMatrixSparseFromDense(float* denseMatrix, std::size_t* bitmap, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type, std::size_t& size);

void transpose(float* matrix, std::size_t rows, std::size_t cols);

void denseToSparse(float* denseMatrix, std::size_t rows, std::size_t cols, std::vector<std::size_t>& rowPointer, std::vector<std::size_t>& colIndex,
                   std::vector<float>& values);

/////
metadata_address_t generateMinorIDFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, std::size_t& nnz, GENERATION_TYPE gen_type);
/////
//int* generateMajorIDFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type);
/////
metadata_address_t generateMajorPointerFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type);

void printDenseMatrix(float* matrix, std::size_t rows, std::size_t cols);

void printBitMap(std::size_t* bitmap, std::size_t rows, std::size_t cols);

void printSparseMatrix(float* sparseMatrix, std::size_t* bitmap, std::size_t rows, std::size_t cols);

float* generatePrunnedMatrix(const float* src_matrix, std::size_t size, float pr_ratio);

//Opt functions
void organizeMatrix(float* matrix, std::size_t M, std::size_t K, std::size_t* pointer_table, GENERATION_TYPE gen_type);

void organizeMatrixBack(float* matrix, std::size_t M, std::size_t K, std::size_t* pointer_table, GENERATION_TYPE gen_type);

std::size_t* calculateOrdering(float* matrix, std::size_t M, std::size_t K, GENERATION_TYPE gen_type, int num_ms);

void sequential_layer(std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X, std::size_t Y,
                      std::size_t strides, float* input, float* filters, float* outputs);

void cpu_gemm(float* MK_dense_matrix, float* KN_dense_matrix, float* output, std::size_t M, std::size_t N, std::size_t K);

#endif  //UTILITY_H
