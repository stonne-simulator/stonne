#ifndef _TESTBENCH_H
#define _TESTBENCH_H
void sequential_layer(std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X, std::size_t Y,
                      std::size_t strides, float* input, float* filters, float* outputs);

void cpu_gemm(float* MK_dense_matrix, float* KN_dense_matrix, float* output, std::size_t M, std::size_t N, std::size_t K);

void run_simple_tests();

void run_stonne_architecture_tests(layerTest layer, std::size_t num_ms);

void hand_tests();

#endif
