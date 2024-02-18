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

constexpr std::size_t T_N = 4;
constexpr std::size_t T_K = 32;

constexpr std::size_t MK_size = M * K;
constexpr std::size_t KN_size = N * K;
constexpr std::size_t outputSize = M * N;

std::vector<float> MK_dense_matrix = genRandom<float>(MK_size, -1, 1);
std::vector<float> KN_dense_matrix = genRandom<float>(KN_size, -1, 1);
std::vector<float> output(outputSize, 0);
std::vector<float> outputCpu(outputSize, 0);

std::vector<std::size_t> MK_col_id;
std::vector<std::size_t> MK_row_pointer;
std::vector<float> MK_sparse_matrix;

Stonne init() {
  prune<float>(MK_dense_matrix, MK_sparsity / 100.0f);

  int nnz = 0;
  std::size_t* p_MK_col_id = reinterpret_cast<std::size_t*>(generateMinorIDFromDense(MK_dense_matrix.data(), M, K, nnz, GEN_BY_ROWS));
  std::size_t* p_MK_row_pointer = reinterpret_cast<std::size_t*>(generateMajorPointerFromDense(MK_dense_matrix.data(), M, K, GEN_BY_ROWS));
  std::size_t MK_sparse_size;
  float* p_MK_sparse_matrix = generateMatrixSparseFromDenseNoBitmap(MK_dense_matrix.data(), M, K, GEN_BY_ROWS, MK_sparse_size);

  // convert arrays to vectors
  MK_col_id = std::vector<std::size_t>(p_MK_col_id, p_MK_col_id + nnz);
  MK_row_pointer = std::vector<std::size_t>(p_MK_row_pointer, p_MK_row_pointer + M + 1);
  MK_sparse_matrix = std::vector<float>(p_MK_sparse_matrix, p_MK_sparse_matrix + nnz);
  delete[] p_MK_col_id;
  delete[] p_MK_row_pointer;
  delete[] p_MK_sparse_matrix;

  Config stonne_cfg;  //Hardware parameters
  stonne_cfg.m_SDMemoryCfg.mem_controller_type = MAGMA_SPARSE_DENSE;
  stonne_cfg.m_MSNetworkCfg.ms_size = 128;
  stonne_cfg.m_SDMemoryCfg.n_read_ports = 64;   // dn_bw
  stonne_cfg.m_SDMemoryCfg.n_write_ports = 64;  // rn_bw
  stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled = true;
  stonne_cfg.print_stats_enabled = false;

  // Preparing the main memory
  Memory<float> main_memory(MK_sparse_size + KN_size + outputSize);
  stonne_cfg.m_SDMemoryCfg.weight_address = 0;
  stonne_cfg.m_SDMemoryCfg.input_address = KN_size;
  stonne_cfg.m_SDMemoryCfg.output_address = KN_size + MK_sparse_size;
  stonne_cfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
  stonne_cfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value
  // Copying the data to the main memory
  std::copy(KN_dense_matrix.begin(), KN_dense_matrix.end(), main_memory.begin());
  std::copy(MK_sparse_matrix.begin(), MK_sparse_matrix.end(), main_memory.begin() + static_cast<std::vector<float>::difference_type>(KN_size));

  Stonne stonne(stonne_cfg, main_memory);
  stonne.loadSparseDense(layer_name, N, K, M, MK_sparse_matrix.data(), KN_dense_matrix.data(), MK_col_id.data(), MK_row_pointer.data(), output.data(), T_N,
                         T_K);

  return stonne;
}

TEST_CASE("SmallSparseDense_MAGMA_Sim", "[sim][test]") {
  Stonne stonne = init();
  stonne.run();

  cpu_gemm(MK_dense_matrix.data(), KN_dense_matrix.data(), outputCpu.data(), M, N, K);
  constexpr float eps = 1e-3;
  REQUIRE(equals(output, outputCpu, eps));
}

TEST_CASE("SmallSparseDense_Profiling", "[sim][benchmark]") {
  BENCHMARK_ADVANCED("STONNE SparseDense Small Benchmark")(Catch::Benchmark::Chronometer meter) {
    Stonne stonne = init();

    meter.measure([&stonne] { stonne.run(); });
    return 0;
  };
}