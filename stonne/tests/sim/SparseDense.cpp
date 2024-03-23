#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "common/utility.hpp"
#include "utils.hpp"

struct TestParams {
  std::size_t M{1}, N{1}, K{1};
  unsigned int MK_sparsity{0};
  Config stonneCfg;
};

struct Test {
  TestParams params;
  std::unique_ptr<Stonne> stonne;
  std::vector<float> A_dense_matrix;
  std::vector<float> B_dense_matrix;
  std::vector<float> output;
  std::vector<float> A_CSR_vals;
  std::vector<std::size_t> A_CSR_rowp;
  std::vector<std::size_t> A_csr_colidx;

  Test(TestParams& p) : params(p) {
    // Layer parameters and generation of input data
    const std::size_t MK_size = params.M * params.K;
    const std::size_t KN_size = params.N * params.K;
    const std::size_t outputSize = params.M * params.N;

    A_dense_matrix = genRandom<float>(MK_size, -1, 1);
    B_dense_matrix = genRandom<float>(KN_size, -1, 1);
    output = std::vector<float>(outputSize, 0);

    prune<float>(A_dense_matrix, params.MK_sparsity / 100.0f);
    A_csr_colidx = generateMinorIDFromDense(A_dense_matrix, params.M, params.K, GEN_BY_ROWS);
    A_CSR_rowp = generateMajorPointerFromDense(A_dense_matrix, params.M, params.K, GEN_BY_ROWS);
    A_CSR_vals = generateMatrixSparseFromDenseNoBitmap(A_dense_matrix, params.M, params.K, GEN_BY_ROWS);

    // MAGMA parameters
    params.stonneCfg.m_MSNetworkCfg.multiplier_network_type = LINEAR;
    params.stonneCfg.m_ASNetworkCfg.reduce_network_type = ASNETWORK;
    params.stonneCfg.m_SDMemoryCfg.mem_controller_type = MAGMA_SPARSE_DENSE;
    params.stonneCfg.m_ASNetworkCfg.accumulation_buffer_enabled = true;

    // Memory configuration
    params.stonneCfg.m_SDMemoryCfg.weight_address = 0;
    params.stonneCfg.m_SDMemoryCfg.input_address = B_dense_matrix.size();
    params.stonneCfg.m_SDMemoryCfg.output_address = B_dense_matrix.size() + A_CSR_vals.size();
    params.stonneCfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
    params.stonneCfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value

    auto memory = std::make_unique<SimpleMem<float>>(A_CSR_vals.size() + B_dense_matrix.size() + output.size());
    memory->fill(0, B_dense_matrix.begin(), B_dense_matrix.end());
    memory->fill(B_dense_matrix.size(), A_CSR_vals.begin(), A_CSR_vals.end());

    // STONNE initialization
    stonne = std::make_unique<Stonne>(params.stonneCfg, std::move(memory));
    stonne->loadSparseDense("Test", params.N, params.K, params.M, A_CSR_vals.data(), B_dense_matrix.data(), A_csr_colidx.data(), A_CSR_rowp.data(),
                            output.data(), 1, 1);
    stonne->generateTile();
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
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 128,
          },
        .m_ASNetworkCfg =
          {
            .accumulation_buffer_enabled = true,
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
    .M = 128,
    .N = 128,
    .K = 256,
    .MK_sparsity = 90,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 128,
          },
        .m_ASNetworkCfg =
          {
            .accumulation_buffer_enabled = true,
          },
        .m_SDMemoryCfg =
          {
            .n_read_ports = 128,   // dn_bw
            .n_write_ports = 128,  // rn_bw
          },
      },
  };
  return {params};
}

TEST_CASE("SmallSparseDense_MAGMA_Sim", "[sim][test][small]") {
  Test t = smallTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 1101);
}

TEST_CASE("LargeSparseDense_MAGMA_Sim", "[sim][test][large]") {
  Test t = largeTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 18049);
}

TEST_CASE("SmallSparseDense_MAGMA_Profiling", "[sim][benchmark][small]") {
  BENCHMARK_ADVANCED("STONNE SparseDense Small Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = smallTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}

TEST_CASE("LargeSparseDense_MAGMA_Profiling", "[sim][benchmark][large]") {
  BENCHMARK_ADVANCED("STONNE SparseDense Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = largeTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}