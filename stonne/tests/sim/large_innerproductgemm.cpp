#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "common/utility.hpp"
#include "utils.hpp"

//Layer parameters
const std::string layer_name = "TestLayer";
constexpr std::size_t M = 256;
constexpr std::size_t N = 128;
constexpr std::size_t K = 256;
constexpr std::size_t MK_sparsity = 80;
constexpr std::size_t KN_sparsity = 50;
constexpr Dataflow dataflow = MK_STA_KN_STR;

constexpr std::size_t MK_size = M * K;
constexpr std::size_t KN_size = N * K;
constexpr std::size_t outputSize = M * N;
std::size_t MK_sparse_size, KN_sparse_size;

std::vector<float> A_dense_matrix;
std::vector<float> B_dense_matrix;
std::vector<float> output(outputSize, 0);
std::vector<std::size_t> outputBitmap(outputSize, 0);
std::vector<float> outputCpu(outputSize, 0);

std::vector<std::size_t> MK_bitmap;
std::vector<std::size_t> KN_bitmap;
std::vector<float> MK_sparse_matrix;
std::vector<float> KN_sparse_matrix;

void init_matrices() {
  A_dense_matrix = genRandom<float>(MK_size, -1, 1);
  B_dense_matrix = genRandom<float>(KN_size, -1, 1);
  prune<float>(A_dense_matrix, MK_sparsity / 100.0f);
  prune<float>(B_dense_matrix, KN_sparsity / 100.0f);

  MK_bitmap = generateBitMapFromDense(A_dense_matrix, M, K, GEN_BY_ROWS);
  KN_bitmap = generateBitMapFromDense(B_dense_matrix, K, N, GEN_BY_COLS);

  MK_sparse_matrix = generateMatrixSparseFromDense(A_dense_matrix, MK_bitmap, M, K, GEN_BY_ROWS, MK_sparse_size);
  KN_sparse_matrix = generateMatrixSparseFromDense(B_dense_matrix, KN_bitmap, K, N, GEN_BY_COLS, KN_sparse_size);
}

Stonne init() {
  Config stonne_cfg = {
      .print_stats_enabled = false,
      .m_MSNetworkCfg = {.ms_size = 128},
      .m_ASNetworkCfg = {.accumulation_buffer_enabled = false},
      .m_SDMemoryCfg = {.mem_controller_type = SIGMA_SPARSE_GEMM, .n_read_ports = 64, .n_write_ports = 64},
  };

  // Preparing the main memory
  auto main_memory = std::make_unique<SimpleMem<float>>(MK_sparse_size + KN_sparse_size + outputSize);
  stonne_cfg.m_SDMemoryCfg.weight_address = 0;
  stonne_cfg.m_SDMemoryCfg.input_address = KN_sparse_size;
  stonne_cfg.m_SDMemoryCfg.output_address = KN_sparse_size + MK_sparse_size;
  stonne_cfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
  stonne_cfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value
  // Copying the data to the main memory
  main_memory->fill(0, KN_sparse_matrix.begin(), KN_sparse_matrix.end());
  main_memory->fill(KN_sparse_size, MK_sparse_matrix.begin(), MK_sparse_matrix.end());

  Stonne stonne(stonne_cfg, std::move(main_memory));
  stonne.loadGEMM(layer_name, N, K, M, MK_sparse_matrix.data(), KN_sparse_matrix.data(), MK_bitmap.data(), KN_bitmap.data(), output.data(), outputBitmap.data(),
                  dataflow);

  return stonne;
}

TEST_CASE("LargeInnerProductGEMM_Flexagon_Sim", "[sim][flexagon][test]") {
  init_matrices();
  Stonne stonne = init();
  stonne.run();

  cpu_gemm(A_dense_matrix.data(), B_dense_matrix.data(), outputCpu.data(), M, N, K);
  constexpr float eps = 1e-3;
  REQUIRE(equals(output, outputCpu, eps));

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(stonne.getNCycles() == 34388);
}

TEST_CASE("LargeInnerProductGEMM_Flexagon_Profiling", "[sim][flexagon][benchmark]") {
  init_matrices();
  BENCHMARK_ADVANCED("STONNE InnerProductGEMM Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Stonne stonne = init();

    meter.measure([&stonne] { stonne.run(); });
    return 0;
  };
}