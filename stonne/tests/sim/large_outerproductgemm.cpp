#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "common/utility.hpp"
#include "utils.hpp"

//Layer parameters
const std::string layer_name = "TestLayer";
constexpr std::size_t M = 1024;
constexpr std::size_t N = 100;
constexpr std::size_t K = 128;
constexpr std::size_t MK_sparsity = 80;
constexpr std::size_t KN_sparsity = 65;

constexpr std::size_t MK_size = M * K;
constexpr std::size_t KN_size = N * K;
constexpr std::size_t outputSize = M * N;

std::vector<float> A_dense_matrix;
std::vector<float> A_T_dense_matrix;
std::vector<float> B_dense_matrix;
std::vector<float> output(outputSize, 0);
std::vector<float> outputCpu(outputSize, 0);

std::vector<std::size_t> MK_CSR_rowp;
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

  // transpose A matrix to obtain it later on CSC
  A_T_dense_matrix = std::vector<float>(A_dense_matrix);
  transpose(A_T_dense_matrix.data(), M, K);

  denseToSparse(A_T_dense_matrix.data(), K, M, MK_CSR_rowp, A_CSR_colidx, A_CSR_vals);
  denseToSparse(B_dense_matrix.data(), K, N, B_CSR_rowp, B_CSR_colidx, B_CSR_vals);
}

Stonne init() {
  Config stonne_cfg = {
      .print_stats_enabled = false,
      .m_MSNetworkCfg = {.multiplier_network_type = SPARSEFLEX_LINEAR, .ms_size = 64},
      .m_ASNetworkCfg = {.reduce_network_type = SPARSEFLEX_MERGER},
      .m_SDMemoryCfg = {.mem_controller_type = OUTER_PRODUCT_GEMM, .n_read_ports = 64, .n_write_ports = 64},
  };

  // Preparing the main memory
  auto main_memory = std::make_unique<SimpleMem<float>>(A_CSR_vals.size() + B_CSR_vals.size() + outputSize);
  stonne_cfg.m_SDMemoryCfg.input_address = 0;
  stonne_cfg.m_SDMemoryCfg.weight_address = A_CSR_vals.size();
  stonne_cfg.m_SDMemoryCfg.output_address = B_CSR_vals.size() + A_CSR_vals.size();
  stonne_cfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
  stonne_cfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value
  // Copying the data to the main memory
  main_memory->fill(0, A_CSR_vals.begin(), A_CSR_vals.end());
  main_memory->fill(A_CSR_vals.size(), B_CSR_vals.begin(), B_CSR_vals.end());

  Stonne stonne(stonne_cfg, std::move(main_memory));
  stonne.loadSparseOuterProduct(layer_name, N, K, M, A_CSR_vals.data(), B_CSR_vals.data(), A_CSR_colidx.data(), MK_CSR_rowp.data(), B_CSR_colidx.data(),
                                B_CSR_rowp.data(), output.data());

  return stonne;
}

TEST_CASE("LargeOuterProductGEMM_Flexagon_Sim", "[sim][flexagon][test]") {
  init_matrices();
  Stonne stonne = init();
  stonne.run();

  cpu_gemm(A_dense_matrix.data(), B_dense_matrix.data(), outputCpu.data(), M, N, K);

  constexpr float eps = 1e-3;
  REQUIRE(equals(output, outputCpu, eps));

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(stonne.getNCycles() == 131784);
}

TEST_CASE("LargeOuterProductGEMM_Flexagon_Profiling", "[sim][flexagon][benchmark]") {
  init_matrices();
  BENCHMARK_ADVANCED("STONNE OuterProductGEMM Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Stonne stonne = init();

    meter.measure([&stonne] { stonne.run(); });
    return 0;
  };
}