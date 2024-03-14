#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "common/utility.hpp"
#include "utils.hpp"

//Layer parameters
const std::string layer_name = "TestLayer";
constexpr std::size_t M = 20;
constexpr std::size_t N = 20;
constexpr std::size_t K = 256;
constexpr std::size_t MK_sparsity = 80;
constexpr std::size_t KN_sparsity = 10;

constexpr std::size_t MK_size = M * K;
constexpr std::size_t KN_size = N * K;
constexpr std::size_t outputSize = M * N;

std::vector<float> A_dense_matrix;
std::vector<float> B_dense_matrix;
std::vector<float> output(outputSize, 0);
std::vector<float> outputCpu(outputSize, 0);

std::vector<std::size_t> A_CSR_rowp;
std::vector<std::size_t> A_CSR_colidx;
std::vector<float> A_CSR_vals;
std::vector<std::size_t> B_CSR_rowp;
std::vector<std::size_t> B_CSR_colidx;
std::vector<float> B_CSR_vals;

void init_matrices() {
  A_dense_matrix = genRandom<float>(MK_size, -1, 1);
  B_dense_matrix = genRandom<float>(KN_size, -1, 1);
  prune<float>(A_dense_matrix, MK_sparsity / 100.0f);
  prune<float>(B_dense_matrix, KN_sparsity / 100.0f);

  denseToSparse(A_dense_matrix.data(), M, K, A_CSR_rowp, A_CSR_colidx, A_CSR_vals);
  denseToSparse(B_dense_matrix.data(), K, N, B_CSR_rowp, B_CSR_colidx, B_CSR_vals);
}

Stonne init() {
  Config stonne_cfg = {
      .print_stats_enabled = false,
      .m_MSNetworkCfg = {.multiplier_network_type = SPARSEFLEX_LINEAR, .ms_size = 16},
      .m_ASNetworkCfg = {.reduce_network_type = SPARSEFLEX_MERGER},
      .m_SDMemoryCfg = {.mem_controller_type = GUSTAVSONS_GEMM, .n_read_ports = 16, .n_write_ports = 16},
  };

  // Preparing the main memory
  Memory<float> main_memory(A_CSR_vals.size() + B_CSR_vals.size() + outputSize);
  stonne_cfg.m_SDMemoryCfg.input_address = 0;
  stonne_cfg.m_SDMemoryCfg.weight_address = A_CSR_vals.size();
  stonne_cfg.m_SDMemoryCfg.output_address = B_CSR_vals.size() + A_CSR_vals.size();
  stonne_cfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
  stonne_cfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value
  // Copying the data to the main memory
  std::copy(A_CSR_vals.begin(), A_CSR_vals.end(), main_memory.begin());
  std::copy(B_CSR_vals.begin(), B_CSR_vals.end(), main_memory.begin() + static_cast<std::vector<float>::difference_type>(A_CSR_vals.size()));

  Stonne stonne(stonne_cfg, main_memory);
  stonne.loadSparseOuterProduct(layer_name, N, K, M, A_CSR_vals.data(), B_CSR_vals.data(), A_CSR_colidx.data(), A_CSR_rowp.data(), B_CSR_colidx.data(),
                                B_CSR_rowp.data(), output.data());

  return stonne;
}

TEST_CASE("SmallGustavsonGEMM_Flexagon_Sim", "[sim][flexagon][test]") {
  init_matrices();
  Stonne stonne = init();
  stonne.run();

  cpu_gemm(A_dense_matrix.data(), B_dense_matrix.data(), outputCpu.data(), M, N, K);

  constexpr float eps = 1e-3;
  REQUIRE(equals(output, outputCpu, eps));
}

TEST_CASE("SmallGustavsonGEMM_Flexagon_Profiling", "[sim][flexagon][benchmark]") {
  init_matrices();
  BENCHMARK_ADVANCED("STONNE SparseGEMM Small Benchmark")(Catch::Benchmark::Chronometer meter) {
    Stonne stonne = init();

    meter.measure([&stonne] { stonne.run(); });
    return 0;
  };
}