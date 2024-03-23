#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "utils.hpp"

struct TestParams {
  std::size_t M{1}, N{1}, K{1};
  Config stonneCfg;
};

struct Test {
  TestParams params;
  std::unique_ptr<Stonne> stonne;
  std::vector<float> A_dense_matrix;
  std::vector<float> B_dense_matrix;
  std::vector<float> output;

  Test(TestParams& p) : params(p) {
    // Layer parameters and generation of input data
    A_dense_matrix = genRandom<float>(params.M * params.K, -1, 1);
    B_dense_matrix = genRandom<float>(params.N * params.K, -1, 1);
    output = std::vector<float>(params.M * params.N, 0);

    // MAERI parameters
    params.stonneCfg.m_MSNetworkCfg.multiplier_network_type = LINEAR;
    params.stonneCfg.m_ASNetworkCfg.reduce_network_type = ASNETWORK;
    params.stonneCfg.m_SDMemoryCfg.mem_controller_type = MAERI_DENSE_WORKLOAD;
    params.stonneCfg.m_ASNetworkCfg.accumulation_buffer_enabled = true;

    // Memory configuration
    params.stonneCfg.m_SDMemoryCfg.weight_address = 0;
    params.stonneCfg.m_SDMemoryCfg.input_address = B_dense_matrix.size();
    params.stonneCfg.m_SDMemoryCfg.output_address = B_dense_matrix.size() + A_dense_matrix.size();
    params.stonneCfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
    params.stonneCfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value

    auto memory = std::make_unique<SimpleMem<float>>(A_dense_matrix.size() + B_dense_matrix.size() + output.size());
    memory->fill(0, B_dense_matrix.begin(), B_dense_matrix.end());
    memory->fill(B_dense_matrix.size(), A_dense_matrix.begin(), A_dense_matrix.end());

    // STONNE initialization
    stonne = std::make_unique<Stonne>(params.stonneCfg, std::move(memory));
    stonne->loadDenseGEMM("Test", params.N, params.K, params.M, A_dense_matrix.data(), B_dense_matrix.data(), output.data(), CNN_DATAFLOW);
    stonne->generateTile();
  }

  void run() { stonne->run(); }

  void check() {
    std::vector<float> outputCpu(output.size(), 0);
    sequential_layer(1, params.K, 1, params.N, 1, params.M, 1, params.K, 1, A_dense_matrix.data(), B_dense_matrix.data(), outputCpu.data());

    constexpr float eps = 1e-3;
    REQUIRE(equals(output, outputCpu, eps));
  }
};

Test smallTest() {
  TestParams params = {
    .M = 20,
    .N = 20,
    .K = 256,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 256,
          },
        .m_SDMemoryCfg =
          {
            .n_read_ports = 256,   // dn_bw
            .n_write_ports = 256,  // rn_bw
          },
      },
  };

  return {params};
}

Test largeTest() {
  TestParams params = {
    .M = 256,
    .N = 128,
    .K = 64,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 256,
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

TEST_CASE("SmallDenseGEMM_MAERI_Sim", "[sim][maeri][test][small]") {
  Test t = smallTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 481);
}

TEST_CASE("LargeDenseGEMM_MAERI_Sim", "[sim][maeri][test][large]") {
  Test t = largeTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 16526);
}

TEST_CASE("SmallDenseGEMM_MAERI_Profiling", "[sim][maeri][benchmark][small]") {
  BENCHMARK_ADVANCED("STONNE DenseGEMM Small Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = smallTest();

    meter.measure([&t] { t.run(); });
    return 0;
  };
}

TEST_CASE("LargeDenseGEMM_MAERI_Profiling", "[sim][maeri][benchmark][large]") {
  BENCHMARK_ADVANCED("STONNE DenseGEMM Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = largeTest();

    meter.measure([&t] { t.run(); });
    return 0;
  };
}