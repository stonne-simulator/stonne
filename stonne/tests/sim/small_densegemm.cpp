#include <catch.hpp>
#include <functional>
#include "STONNEModel.hpp"
#include "utils.hpp"

//Layer parameters
const std::string layer_name = "TestLayer";
constexpr std::size_t M = 20;
constexpr std::size_t N = 20;
constexpr std::size_t K = 256;

constexpr std::size_t T_M = 2;
constexpr std::size_t T_N = 1;
constexpr std::size_t T_K = 64;

constexpr std::size_t MK_size = M * K;
constexpr std::size_t KN_size = N * K;
constexpr std::size_t outputSize = M * N;

std::vector<float> MK_dense_matrix = genRandom<float>(MK_size, -1, 1);
std::vector<float> KN_dense_matrix = genRandom<float>(KN_size, -1, 1);
std::vector<float> output(outputSize, 0);
std::vector<float> outputCpu(outputSize, 0);

Stonne init() {
  Config stonne_cfg;  //Hardware parameters
  stonne_cfg.m_MSNetworkCfg.ms_size = 256;
  stonne_cfg.m_SDMemoryCfg.n_read_ports = 64;   // dn_bw
  stonne_cfg.m_SDMemoryCfg.n_write_ports = 64;  // rn_bw
  stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled = false;
  stonne_cfg.print_stats_enabled = false;

  // Preparing the main memory
  Memory<float> main_memory(MK_size + KN_size + outputSize);
  stonne_cfg.m_SDMemoryCfg.weight_address = 0;
  stonne_cfg.m_SDMemoryCfg.input_address = KN_size;
  stonne_cfg.m_SDMemoryCfg.output_address = KN_size + MK_size;
  stonne_cfg.m_SDMemoryCfg.data_width = 1;     // TODO IMPORTANT: this is only used for STONNE fake memory
  stonne_cfg.m_SDMemoryCfg.n_write_mshr = 16;  // default value
  // Copying the data to the main memory
  std::copy(KN_dense_matrix.begin(), KN_dense_matrix.end(), main_memory.begin());
  std::copy(MK_dense_matrix.begin(), MK_dense_matrix.end(), main_memory.begin() + static_cast<std::vector<float>::difference_type>(KN_size));

  Stonne stonne(stonne_cfg, main_memory);
  stonne.loadDenseGEMM(layer_name, N, K, M, MK_dense_matrix.data(), KN_dense_matrix.data(), output.data(), CNN_DATAFLOW);  //Loading the layer
  stonne.loadGEMMTile(T_N, T_K, T_M);

  return stonne;
}

TEST_CASE("SmallDenseGEMM_MAERI_Sim", "[sim][test]") {
  Stonne stonne = init();
  stonne.run();

  sequential_layer(1, K, 1, N, 1, M, 1, K, 1, MK_dense_matrix.data(), KN_dense_matrix.data(), outputCpu.data());
  constexpr float eps = 1e-3;
  REQUIRE(equals(output, outputCpu, eps));
}

TEST_CASE("SmallDenseGEMM_Profiling", "[sim][benchmark]") {
  BENCHMARK_ADVANCED("STONNE DenseGEMM Benchmark")(Catch::Benchmark::Chronometer meter) {
    Stonne stonne = init();

    meter.measure([&stonne] { stonne.run(); });
    return 0;
  };
}