#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "utils.hpp"

struct TestParams {
  std::size_t R{1}, S{1}, C{1}, G{1}, K{1}, N{1}, X{1}, Y{1}, stride{1};
  Config stonneCfg;
};

struct Test {
  TestParams params;
  std::unique_ptr<Stonne> stonne;
  std::vector<float> ifmap;
  std::vector<float> filter;
  std::vector<float> ofmap;

  Test(TestParams& p) : params(p) {
    // Layer parameters and generation of input data
    const std::size_t X_ = (params.X - params.R + params.stride) / params.stride;
    const std::size_t Y_ = (params.Y - params.S + params.stride) / params.stride;

    ifmap = genRandom<float>(params.N * params.X * params.Y * params.C, -1, 1);
    filter = genRandom<float>(params.R * params.S * (params.C / params.G) * params.K, -1, 1);
    ofmap = std::vector<float>(params.N * X_ * Y_ * params.K, 0);

    // MAERI parameters
    params.stonneCfg.m_MSNetworkCfg.multiplier_network_type = LINEAR;
    params.stonneCfg.m_ASNetworkCfg.reduce_network_type = ASNETWORK;
    params.stonneCfg.m_SDMemoryCfg.mem_controller_type = MAERI_DENSE_WORKLOAD;
    params.stonneCfg.m_ASNetworkCfg.accumulation_buffer_enabled = true;

    // Memory configuration
    params.stonneCfg.m_SDMemoryCfg.weight_address = 0;
    params.stonneCfg.m_SDMemoryCfg.input_address = filter.size();
    params.stonneCfg.m_SDMemoryCfg.output_address = filter.size() + ifmap.size();
    params.stonneCfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
    params.stonneCfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value

    auto memory = std::make_unique<SimpleMem<float>>(ifmap.size() + filter.size() + ofmap.size());
    memory->fill(0, filter.begin(), filter.end());
    memory->fill(filter.size(), ifmap.begin(), ifmap.end());

    // STONNE initialization
    stonne = std::make_unique<Stonne>(params.stonneCfg, std::move(memory));
    stonne->loadDNNLayer(CONV, "Test", params.R, params.S, params.C, params.K, params.G, params.N, params.X, params.Y, params.stride, ifmap.data(),
                         filter.data(), ofmap.data(), CNN_DATAFLOW);
    stonne->generateTile();
  }

  void run() { stonne->run(); }

  void check() {
    std::vector<float> ofmapCpu(ofmap.size(), 0);
    sequential_layer(params.R, params.S, params.C, params.K, params.G, params.N, params.X, params.Y, params.stride, ifmap.data(), filter.data(),
                     ofmapCpu.data());

    constexpr float eps = 1e-3;
    REQUIRE(equals(ofmap, ofmapCpu, eps));
  }
};

Test smallTest() {
  TestParams params = {
    .R = 3,
    .S = 3,
    .C = 6,
    .G = 1,
    .K = 6,
    .N = 1,
    .X = 20,
    .Y = 20,
    .stride = 1,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 64,
          },
        .m_SDMemoryCfg =
          {
            .n_read_ports = 8,   // dn_bw
            .n_write_ports = 8,  // rn_bw
          },
      },
  };

  return {params};
}

Test largeTest() {
  TestParams params = {
    .R = 5,
    .S = 5,
    .C = 7,
    .G = 1,
    .K = 6,
    .N = 4,
    .X = 30,
    .Y = 30,
    .stride = 1,
    .stonneCfg =
      {
        .print_stats_enabled = false,
        .m_MSNetworkCfg =
          {
            .ms_size = 64,
          },
        .m_ASNetworkCfg =
          {
            .accumulation_buffer_enabled = false,
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

TEST_CASE("SmallCONV_MAERI_Sim", "[sim][maeri][test][small]") {
  Test t = smallTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 6487);
}

TEST_CASE("LargeCONV_MAERI_Sim", "[sim][maeri][test][large]") {
  Test t = largeTest();
  t.run();
  t.check();

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(t.stonne->getNCycles() == 56899);
}

TEST_CASE("SmallCONV_MAERI_Profiling", "[sim][maeri][benchmark][small]") {
  BENCHMARK_ADVANCED("STONNE CONV Small Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = smallTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}

TEST_CASE("LargeCONV_MAERI_Profiling", "[sim][maeri][benchmark][large]") {
  BENCHMARK_ADVANCED("STONNE CONV Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Test t = largeTest();
    meter.measure([&t] { t.run(); });
    return 0;
  };
}