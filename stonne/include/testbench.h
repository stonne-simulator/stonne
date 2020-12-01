#ifndef _TESTBENCH_H
#define _TESTBENCH_H
void sequential_layer(unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides,
float* input, float* filters, float * outputs);

void cpu_gemm(float* MK_dense_matrix, float* KN_dense_matrix, float* output, unsigned int M, unsigned int N, unsigned int K);


void run_simple_tests();

void run_stonne_architecture_tests(layerTest layer, unsigned int num_ms);

void hand_tests();

#endif
