#ifndef STONNEMODEL_H_
#define STONNEMODEL_H_

#include <string>
#include "comm/CollectionBus.hpp"
#include "comm/Connection.hpp"
#include "common/Config.hpp"
#include "common/types.hpp"
#include "common/utility.hpp"
#include "compiler/CompilerART.hpp"
#include "compiler/CompilerMSN.hpp"
#include "memctrl/GustavsonsSpGEMMSDMemory.hpp"
#include "memctrl/MemoryController.hpp"
#include "memctrl/OSMeshMN.hpp"
#include "memctrl/OSMeshSDMemory.hpp"
#include "memctrl/OuterLoopSpGEMMSDMemory.hpp"
#include "memctrl/SDMemory.hpp"
#include "memctrl/SparseDenseSDMemory.hpp"
#include "memctrl/SparseSDMemory.hpp"
#include "memory/LookupTable.hpp"
#include "memory/Memory.hpp"
#include "network/dn/DSNetworkTop.hpp"
#include "network/dn/DistributionNetwork.hpp"
#include "network/mn/MSNetwork.hpp"
#include "network/mn/SparseFlex_MSNetwork.hpp"
#include "network/rn/ASNetwork.hpp"
#include "network/rn/FENetwork.hpp"
#include "network/rn/ReduceNetwork.hpp"
#include "network/rn/SparseFlex_ASNetwork.hpp"
#include "network/rn/TemporalRN.hpp"
#include "tile_generator/TileGenerator.hpp"

class Stonne {
 private:
  //Hardware paramenters
  Config stonne_cfg;
  std::size_t ms_size;         //Number of multipliers
  std::size_t n_adders;        //Number of adders obtained from ms_size
  DistributionNetwork* dsnet;  //Distribution Network
  MultiplierNetwork* msnet;    //Multiplier Network
  ReduceNetwork* asnet;        //ART Network
  LookupTable* lt;             //Lookuptable
  MemoryController* mem;       //MemoryController abstraction (e.g., SDMemory from MAERI)
  Bus* collectionBusRN;        //CollectionBus
  Bus* collectionBusMN;
  Connection* outputASConnection;     //The last connection of the AS and input to the lookuptable
  Connection* outputLTConnection;     //Output of the lookup table connection and write port to the SDMemory
  Connection** addersBusConnections;  //Array of output connections between the adders and the bus
  Connection** BusMemoryConnections;  //Array of output Connections between the bus and the memory. (Write output ports)

  //Software parameters
  DNNLayer* dnn_layer;
  Tile* current_tile;
  bool layer_loaded;  //Indicates if the function loadDNN
  bool tile_loaded;

  //Connection and cycle functions
  void connectMemoryandDSN();
  void connectMSNandDSN();  //Function to connect the multiplieers of the MSN to the last level switches in the DSN.
  void connectMSNandASN();
  void connectASNandBus();  //Connect the adders to the Collection bus
  void connectMSNandBus();
  void connectBusandMemory();  //Connect the bus and the memory write ports.
  void printGlobalStats(std::ofstream& out, std::size_t indent);

  // DEBUG PARAMETERS
  unsigned long time_ds;
  unsigned long time_ms;
  unsigned long time_as;
  unsigned long time_lt;
  unsigned long time_mem;
  //DEBUG functions
  void testDSNetwork(std::size_t num_ms);
  void testTile(std::size_t num_ms);
  void testMemory(std::size_t num_ms);

  //Statistics
  std::size_t n_cycles;

  //SST variables and structures
  Memory<float> memHierarchy;

 public:
  Stonne(Config stonne_cfg, Memory<float> mem);
  ~Stonne();

  void loadDNNLayer(Layer_t layer_type, std::string layer_name, std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N,
                    std::size_t X, std::size_t Y, std::size_t strides, address_t input_address, address_t filter_address, address_t output_address,
                    Dataflow dataflow);  //General constructor

  //Load CONV Layer. At the end this calls to the general constructor  with all the parameters
  void loadCONVLayer(std::string layer_name, std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X,
                     std::size_t Y, std::size_t strides, address_t input_address, address_t filter_address, address_t output_address);

  //Load FC layer just with the appropiate parameters
  //N = batch size (i.e., number of rows in input matrix); S=number of inputs per batch (i.e., column size in input matrix and weight matrix); K=number of outputs neurons (i.e, number of rows weight matrix)
  void loadFCLayer(std::string layer_name, std::size_t N, std::size_t S, std::size_t K, address_t input_address, address_t filter_address,
                   address_t output_address);

  //Load Sparse GEMM onto STONNE according to SIGMA parameter taxonomy.
  void loadGEMM(std::string layer_name, std::size_t N, std::size_t K, std::size_t M, address_t MK_matrix, address_t KN_matrix, metadata_address_t MK_metadata,
                metadata_address_t KN_metadata, address_t output_matrix, metadata_address_t output_metadata, Dataflow dataflow);

  //Load Dense GEMM onto STONNE according to SIGMA parameter taxonomy and tiling according to T_N, T_K and T_M
  void loadDenseGEMM(std::string layer_name, std::size_t N, std::size_t K, std::size_t M, address_t MK_matrix, address_t KN_matrix, address_t output_matrix,
                     Dataflow dataflow);

  //Load sparse-dense GEMM onto STONNE
  void loadSparseDense(std::string layer_name, std::size_t N, std::size_t K, std::size_t M, address_t MK_matrix, address_t KN_matrix,
                       metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, address_t output_matrix, std::size_t T_N, std::size_t T_K);

  void loadSparseOuterProduct(std::string layer_name, std::size_t N, std::size_t K, std::size_t M, address_t MK_matrix, address_t KN_matrix,
                              metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, metadata_address_t KN_metadata_id,
                              metadata_address_t KN_metadata_pointer, address_t output_matrix);

  //Load a Dense GEMM tile to run it using the loadDenseGEMM function
  void loadGEMMTile(std::size_t T_N, std::size_t T_K, std::size_t T_M);

  void loadTile(std::size_t T_R, std::size_t T_S, std::size_t T_C, std::size_t T_K, std::size_t T_G, std::size_t T_N, std::size_t T_X_,
                std::size_t T_Y_);  //Load general and CONV tile

  // Loads a FC tile to run it using the loadFC function
  void loadFCTile(std::size_t T_S, std::size_t T_N, std::size_t T_K);  //VNSize = T_S, NumVNs= T_N*T_K

  // Loads a SparseDense tile to run it using the loadSparseDense function
  void loadSparseDenseTile(std::size_t T_N, std::size_t T_K);

  //Loads a tile configuration from a file
  void loadTile(std::string tile_file);

  // Generates a tile configuration using a TileGenerator module
  void generateTile(TileGenerator::Generator generator = TileGenerator::Generator::CHOOSE_AUTOMATICALLY,
                    TileGenerator::Target target = TileGenerator::Target::PERFORMANCE, float MK_sparsity = 0.0f);

  void run();
  void cycle();
  void printStats();
  void printEnergy();
  bool isExecutionFinished();
};

#endif
//TO DO add enumerate.
