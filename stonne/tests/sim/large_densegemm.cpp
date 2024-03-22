#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "utils.hpp"

//Layer parameters
const std::string layer_name = "TestLayer";
constexpr std::size_t M = 256;
constexpr std::size_t N = 150;
constexpr std::size_t K = 64;

constexpr std::size_t T_M = 4;
constexpr std::size_t T_N = 1;
constexpr std::size_t T_K = 64;

constexpr std::size_t MK_size = M * K;
constexpr std::size_t KN_size = N * K;
constexpr std::size_t outputSize = M * N;

std::vector<float> A_dense_matrix = genRandom<float>(MK_size, -1, 1);
std::vector<float> B_dense_matrix = genRandom<float>(KN_size, -1, 1);
std::vector<float> output(outputSize, 0);
std::vector<float> outputCpu(outputSize, 0);

Stonne init() {
  Config stonne_cfg;  //Hardware parameters
  stonne_cfg.m_MSNetworkCfg.ms_size = 256;
  stonne_cfg.m_SDMemoryCfg.n_read_ports = 256;   // dn_bw
  stonne_cfg.m_SDMemoryCfg.n_write_ports = 256;  // rn_bw
  stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled = true;
  stonne_cfg.print_stats_enabled = false;

  // Preparing the main memory
  auto main_memory = std::make_unique<SimpleMem<float>>(MK_size + KN_size + outputSize);
  stonne_cfg.m_SDMemoryCfg.weight_address = 0;
  stonne_cfg.m_SDMemoryCfg.input_address = KN_size;
  stonne_cfg.m_SDMemoryCfg.output_address = KN_size + MK_size;
  stonne_cfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
  stonne_cfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value
  // Copying the data to the main memory
  main_memory->fill(0, B_dense_matrix.begin(), B_dense_matrix.end());
  main_memory->fill(KN_size, A_dense_matrix.begin(), A_dense_matrix.end());

  Stonne stonne(stonne_cfg, std::move(main_memory));
  stonne.loadDenseGEMM(layer_name, N, K, M, A_dense_matrix.data(), B_dense_matrix.data(), output.data(), CNN_DATAFLOW);  //Loading the layer
  stonne.loadGEMMTile(T_N, T_K, T_M);

  return stonne;
}

TEST_CASE("LargeDenseGEMM_MAERI_Sim", "[sim][maeri][test]") {
  Stonne stonne = init();
  stonne.run();

  sequential_layer(1, K, 1, N, 1, M, 1, K, 1, A_dense_matrix.data(), B_dense_matrix.data(), outputCpu.data());
  constexpr float eps = 1e-3;
  REQUIRE(equals(output, outputCpu, eps));

  // Temporal check to ensure that I don't introduce errors during the refactor
  REQUIRE(stonne.getNCycles() == 21001);
}

TEST_CASE("LargeDenseGEMM_MAERI_Profiling", "[sim][maeri][benchmark]") {
  BENCHMARK_ADVANCED("STONNE DenseGEMM Large Benchmark")(Catch::Benchmark::Chronometer meter) {
    Stonne stonne = init();

    meter.measure([&stonne] { stonne.run(); });
    return 0;
  };
}