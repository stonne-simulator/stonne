#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "utils.hpp"

//Layer parameters
const std::string layer_name = "TestLayer";
constexpr std::size_t R = 5;
constexpr std::size_t S = 5;
constexpr std::size_t C = 7;
constexpr std::size_t G = 1;
constexpr std::size_t K = 6;
constexpr std::size_t N = 4;
constexpr std::size_t X = 30;
constexpr std::size_t Y = 30;

constexpr std::size_t T_R = 5;
constexpr std::size_t T_S = 5;
constexpr std::size_t T_C = 1;
constexpr std::size_t T_G = 1;
constexpr std::size_t T_K = 2;
constexpr std::size_t T_N = 1;
constexpr std::size_t T_X_ = 1;
constexpr std::size_t T_Y_ = 1;

constexpr std::size_t strides = 1;
constexpr std::size_t X_ = (X - R + strides) / strides;  // X_
constexpr std::size_t Y_ = (Y - S + strides) / strides;  // Y_

//Creating arrays to store the ifmap ofmap and weights
constexpr std::size_t ifmap_size = N * X * Y * C;
constexpr std::size_t filter_size = R * S * (C / G) * K;
constexpr std::size_t ofmap_size = N * X_ * Y_ * K;

// create vectors and initialize them with random values between -10 and 10
std::vector<float> ifmap = genRandom<float>(ifmap_size, -1, 1);
std::vector<float> filter = genRandom<float>(filter_size, -1, 1);
std::vector<float> ofmap(ofmap_size, 0);
std::vector<float> ofmap_cpu(ofmap_size, 0);

Stonne init() {
  Config stonne_cfg;  //Hardware parameters
  stonne_cfg.m_MSNetworkCfg.ms_size = 64;
  stonne_cfg.m_SDMemoryCfg.n_read_ports = 64;   // dn_bw
  stonne_cfg.m_SDMemoryCfg.n_write_ports = 64;  // rn_bw
  stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled = false;
  stonne_cfg.print_stats_enabled = false;

  // Preparing the main memory
  auto main_memory = std::make_unique<SimpleMem<float>>(ifmap_size + filter_size + ofmap_size);
  stonne_cfg.m_SDMemoryCfg.weight_address = 0;
  stonne_cfg.m_SDMemoryCfg.input_address = filter_size;
  stonne_cfg.m_SDMemoryCfg.output_address = filter_size + ifmap_size;
  stonne_cfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
  stonne_cfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value
  // Copying the data to the main memory
  main_memory->fill(0, filter.begin(), filter.end());
  main_memory->fill(filter_size, ifmap.begin(), ifmap.end());

  Stonne stonne(stonne_cfg, std::move(main_memory));
  stonne.loadDNNLayer(CONV, layer_name, R, S, C, K, G, N, X, Y, strides, ifmap.data(), filter.data(), ofmap.data(), CNN_DATAFLOW);
  stonne.loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_);

  return stonne;
}

TEST_CASE("LargeCONV_MAERI_Sim", "[sim][maeri][test]") {
  Stonne stonne = init();
  stonne.run();

  sequential_layer(R, S, C, K, G, N, X, Y, strides, ifmap.data(), filter.data(), ofmap_cpu.data());
  constexpr float eps = 1e-3;
  REQUIRE(equals(ofmap, ofmap_cpu, eps));

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(stonne.getNCycles() == 398441);
}

TEST_CASE("LargeCONV_MAERI_Profiling", "[sim][maeri][benchmark]") {
  BENCHMARK_ADVANCED("STONNE CONV Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Stonne stonne = init();

    meter.measure([&stonne] { stonne.run(); });
    return 0;
  };
}