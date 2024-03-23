#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "common/utility.hpp"
#include "utils.hpp"

struct TestParams {
  std::size_t M{1}, N{1}, K{1};
  unsigned int MK_sparsity{0}, KN_sparsity{0};
  Config stonneCfg;
};

struct Test {
  TestParams params;
  std::unique_ptr<Stonne> stonne;
  std::vector<float> A_dense_matrix;
  std::vector<float> B_dense_matrix;
  std::vector<float> output;
  std::vector<float> MK_sparse_matrix;
  std::vector<std::size_t> MK_bitmap;
  std::vector<float> KN_sparse_matrix;
  std::vector<std::size_t> KN_bitmap;
  std::vector<std::size_t> outputBitmap;

  Test(TestParams& p) : params(p) {
    // Layer parameters and generation of input data
    A_dense_matrix = genRandom<float>(params.M * params.K, -1, 1);
    B_dense_matrix = genRandom<float>(params.N * params.K, -1, 1);
    output = std::vector<float>(params.M * params.N, 0);

    prune<float>(A_dense_matrix, params.MK_sparsity / 100.0f);
    prune<float>(B_dense_matrix, params.KN_sparsity / 100.0f);

    MK_bitmap = generateBitMapFromDense(A_dense_matrix, params.M, params.K, GEN_BY_ROWS);
    KN_bitmap = generateBitMapFromDense(B_dense_matrix, params.K, params.N, GEN_BY_COLS);
    MK_sparse_matrix = generateMatrixSparseFromDense(A_dense_matrix, MK_bitmap, params.M, params.K, GEN_BY_ROWS);
    KN_sparse_matrix = generateMatrixSparseFromDense(B_dense_matrix, KN_bitmap, params.K, params.N, GEN_BY_COLS);

    // SIGMA parameters
    params.stonneCfg.m_MSNetworkCfg.multiplier_network_type = LINEAR;
    params.stonneCfg.m_ASNetworkCfg.reduce_network_type = ASNETWORK;
    params.stonneCfg.m_SDMemoryCfg.mem_controller_type = SIGMA_SPARSE_GEMM;
    params.stonneCfg.m_ASNetworkCfg.accumulation_buffer_enabled = false;

    // Memory configuration
    params.stonneCfg.m_SDMemoryCfg.weight_address = 0;
    params.stonneCfg.m_SDMemoryCfg.input_address = KN_sparse_matrix.size();
    params.stonneCfg.m_SDMemoryCfg.output_address = KN_sparse_matrix.size() + MK_sparse_matrix.size();
    params.stonneCfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
    params.stonneCfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value

    auto memory = std::make_unique<SimpleMem<float>>(MK_sparse_matrix.size() + KN_sparse_matrix.size() + output.size());
    memory->fill(0, KN_sparse_matrix.begin(), KN_sparse_matrix.end());
    memory->fill(KN_sparse_matrix.size(), MK_sparse_matrix.begin(), MK_sparse_matrix.end());

    // STONNE initialization
    stonne = std::make_unique<Stonne>(params.stonneCfg, std::move(memory));
    stonne->loadGEMM("Test", params.N, params.K, params.M, MK_sparse_matrix.data(), KN_sparse_matrix.data(), MK_bitmap.data(), KN_bitmap.data(), output.data(),
                     outputBitmap.data(), MK_STA_KN_STR);
  }

  void run() { stonne->run(); }

  void check() {
    std::vector<float> outputCpu(output.size(), 0);
    cpu_gemm(A_dense_matrix.data(), B_dense_matrix.data(), outputCpu.data(), params.M, params.N, params.K);

    constexpr float eps = 1e-3;
    REQUIRE(equals(output, outputCpu, eps));
  }
};

Test smallTest() {
  TestParams params = {
    .M = 20,
    .N = 20,
    .K = 256,
    .MK_sparsity = 80,
    .KN_sparsity = 10,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 128,
          },
        .m_SDMemoryCfg =
          {
            .n_read_ports = 64,   // dn_bw
            .n_write_ports = 64,  // rn_bw
          },
      },
  };

  return {params};
}

Test largeTest() {
  TestParams params = {
    .M = 256,
    .N = 128,
    .K = 256,
    .MK_sparsity = 80,
    .KN_sparsity = 50,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 128,
          },
        .m_SDMemoryCfg =
          {
            .n_read_ports = 64,   // dn_bw
            .n_write_ports = 64,  // rn_bw
          },
      },
  };

  return {params};
}

TEST_CASE("SmallInnerProductGEMM_Flexagon_Sim", "[sim][flexagon][test][small]") {
  Test t = smallTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 547);
}

TEST_CASE("LargeInnerProductGEMM_Flexagon_Sim", "[sim][flexagon][test][large]") {
  Test t = largeTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 34388);
}

TEST_CASE("SmallInnerProductGEMM_Flexagon_Profiling", "[sim][flexagon][benchmark][small]") {
  BENCHMARK_ADVANCED("STONNE InnerProductGEMM Small Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = smallTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}

TEST_CASE("LargeInnerProductGEMM_Flexagon_Profiling", "[sim][flexagon][benchmark][large]") {
  BENCHMARK_ADVANCED("STONNE InnerProductGEMM Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = largeTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}