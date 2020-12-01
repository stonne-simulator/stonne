
#ifndef UTILITY_H_
#define UTILITY_H_

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "types.h"

bool isNum(std::string str);

bool ispowerof2(unsigned int x);

std::string getstr(std::istringstream& instr);

std::string get_string_adder_configuration(adderconfig_t config);

std::string get_string_fwlink_direction(fl_t fl_direction);

std::string ind(unsigned int indent); //Get a string with as many spaces as indent value.

std::string get_string_reduce_network_type(ReduceNetwork_t reduce_network_type);

ReduceNetwork_t get_type_reduce_network_type(std::string reduce_network_type);

MemoryController_t get_type_memory_controller_type(std::string memory_controller_type);

std::string get_string_memory_controller_type(MemoryController_t memory_controller_type);

std::string get_string_multiplier_network_type(MultiplierNetwork_t multiplier_network_type);

MultiplierNetwork_t get_type_multiplier_network_type(std::string multiplier_network_type);

Dataflow get_type_dataflow_type(std::string dataflow_type);

std::string get_string_dataflow_type(Dataflow dataflow);

float* generateMatrixDense(unsigned int rows, unsigned int cols, unsigned int sparsity);

unsigned int* generateBitMapFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type);


float* generateMatrixSparseFromDenseNoBitmap(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type) ;
float* generateMatrixSparseFromDense(float* denseMatrix, unsigned int* bitmap, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type);

/////
int* generateMinorIDFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, int &nnz, GENERATION_TYPE gen_type);
/////
//int* generateMajorIDFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type);
/////
int* generateMajorPointerFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type);

void printDenseMatrix(float* matrix, unsigned int rows, unsigned int cols);

void printBitMap(unsigned int* bitmap, unsigned int rows, unsigned int cols);

void printSparseMatrix(float* sparseMatrix, unsigned int* bitmap, unsigned int rows, unsigned int cols);

float* generatePrunnedMatrix(const float* src_matrix, unsigned int size, float pr_ratio);

//Opt functions
void organizeMatrix (float* matrix, unsigned int M, unsigned int K, unsigned int* pointer_table, GENERATION_TYPE gen_type);

void organizeMatrixBack (float* matrix, unsigned int M, unsigned int K, unsigned int* pointer_table, GENERATION_TYPE gen_type);

unsigned int* calculateOrdering (float* matrix, unsigned int M, unsigned int K, GENERATION_TYPE gen_type, int num_ms);

#endif //UTILITY_H
