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
  std::vector<float> A_T_dense_matrix;
  std::vector<float> B_dense_matrix;
  std::vector<float> output;
  std::vector<float> A_CSC_vals;
  std::vector<std::size_t> A_CSC_colp;
  std::vector<std::size_t> A_CSC_rowidx;
  std::vector<float> B_CSR_vals;
  std::vector<std::size_t> B_CSR_rowp;
  std::vector<std::size_t> B_CSR_colidx;

  Test(TestParams& p) : params(p) {
    // Layer parameters and generation of input data
    A_dense_matrix = genRandom<float>(params.M * params.K, -1, 1);
    B_dense_matrix = genRandom<float>(params.N * params.K, -1, 1);
    output = std::vector<float>(params.M * params.N, 0);

    prune<float>(A_dense_matrix, params.MK_sparsity / 100.0f);
    prune<float>(B_dense_matrix, params.KN_sparsity / 100.0f);

    A_T_dense_matrix = std::vector<float>(A_dense_matrix);
    transpose(A_T_dense_matrix.data(), params.M, params.K);

    denseToSparse(A_T_dense_matrix.data(), params.K, params.M, A_CSC_colp, A_CSC_rowidx, A_CSC_vals);
    denseToSparse(B_dense_matrix.data(), params.K, params.N, B_CSR_rowp, B_CSR_colidx, B_CSR_vals);

    // SIGMA parameters
    params.stonneCfg.m_MSNetworkCfg.multiplier_network_type = SPARSEFLEX_LINEAR;
    params.stonneCfg.m_ASNetworkCfg.reduce_network_type = SPARSEFLEX_MERGER;
    params.stonneCfg.m_SDMemoryCfg.mem_controller_type = OUTER_PRODUCT_GEMM;
    params.stonneCfg.m_ASNetworkCfg.accumulation_buffer_enabled = false;

    // Memory configuration
    params.stonneCfg.m_SDMemoryCfg.input_address = 0;
    params.stonneCfg.m_SDMemoryCfg.weight_address = A_CSC_vals.size();
    params.stonneCfg.m_SDMemoryCfg.output_address = B_CSR_vals.size() + A_CSC_vals.size();
    params.stonneCfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
    params.stonneCfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value

    auto main_memory = std::make_unique<SimpleMem<float>>(A_CSC_vals.size() + B_CSR_vals.size() + output.size());
    main_memory->fill(0, A_CSC_vals.begin(), A_CSC_vals.end());
    main_memory->fill(A_CSC_vals.size(), B_CSR_vals.begin(), B_CSR_vals.end());

    // STONNE initialization
    stonne = std::make_unique<Stonne>(params.stonneCfg, std::move(main_memory));
    stonne->loadSparseOuterProduct("Test", params.N, params.K, params.M, A_CSC_vals.data(), B_CSR_vals.data(), A_CSC_rowidx.data(), A_CSC_colp.data(),
                                   B_CSR_colidx.data(), B_CSR_rowp.data(), output.data());
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
            .ms_size = 16,
          },
        .m_SDMemoryCfg =
          {
            .n_read_ports = 16,   // dn_bw
            .n_write_ports = 16,  // rn_bw
          },
      },
  };

  return {params};
}

Test largeTest() {
  TestParams params = {
    .M = 1024,
    .N = 100,
    .K = 128,
    .MK_sparsity = 80,
    .KN_sparsity = 65,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 64,
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

TEST_CASE("SmallOuterProductGEMM_Flexagon_Sim", "[sim][flexagon][test][small]") {
  Test t = smallTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 4541);
}

TEST_CASE("LargeOuterProductGEMM_Flexagon_Sim", "[sim][flexagon][test][large]") {
  Test t = largeTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 131784);
}

TEST_CASE("SmallOuterProductGEMM_Flexagon_Profiling", "[sim][flexagon][benchmark][small]") {
  BENCHMARK_ADVANCED("STONNE OuterProductGEMM Small Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = smallTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}

TEST_CASE("LargeOuterProductGEMM_Flexagon_Profiling", "[sim][flexagon][benchmark][large]") {
  BENCHMARK_ADVANCED("STONNE OuterProductGEMM Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = largeTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}