#ifndef __SPARSEDENSESDMEMORY__H__
#define __SPARSEDENSESDMEMORY__H__

#include <list>
#include "MemoryController.hpp"
#include "comm/Connection.hpp"
#include "comm/DataPackage.hpp"
#include "common/Config.hpp"
#include "common/Fifo.hpp"
#include "common/Stats.hpp"
#include "common/Unit.hpp"
#include "common/dnn/DNNLayer.hpp"
#include "common/dnn/Tile.hpp"
#include "common/types.hpp"
#include "memory/Memory.hpp"
#include "network/mn/MultiplierNetwork.hpp"
#include "network/rn/ReduceNetwork.hpp"

class SparseDenseSDMemory : public MemoryController {
 private:
  DNNLayer* dnn_layer;                    // Layer loaded in the accelerator
  ReduceNetwork* reduce_network;          //Reduce network used to be reconfigured
  MultiplierNetwork* multiplier_network;  //Multiplier network used to be reconfigured

  std::size_t M;
  std::size_t N;

  std::size_t dim_sta;          //Number of vectors sta matrix. Extracted from dnn_layer->get_K(); (See equivalence with CNN)
  std::size_t K;                //Number of columns MK matrix and rows KN matrix. Extracted from dnn_layer->get_C();
  std::size_t dim_str;          //Number of vectors str matrix. Extracted from dnn_layer->get_N()
  std::size_t STA_DIST_ELEM;    //Distance in bitmap memory between two elements of the same vector
  std::size_t STA_DIST_VECTOR;  //Disctance in bitmap memory between two elements of differ vectors.

  std::size_t STR_DIST_ELEM;  //Idem than before but with the STR matrix
  std::size_t STR_DIST_VECTOR;

  std::size_t OUT_DIST_VN;            //To calculate the output memory address
  std::size_t OUT_DIST_VN_ITERATION;  //To calculate the memory address
  Connection* write_connection;
  SparsityControllerState current_state;      //Stage to control what to do according to the state
  std::vector<SparseVN> configurationVNs;     //A set of each VN size mapped onto the architecture.
  std::vector<std::size_t> vnat_table_itern;  //Every element is a VN, indicating the column that is calculating
  std::vector<std::size_t> vnat_table_iterm;  //Iden but for row
  //Connection* read_connection;
  std::vector<Connection*> read_connections;  //Input port connections. There are as many connections as n_read_ports are specified.

  //Input parameters
  std::size_t num_ms;
  std::size_t n_read_ports;
  std::size_t n_write_ports;
  std::size_t write_buffer_capacity;
  std::size_t port_width;

  std::size_t ms_size_per_input_port;
  //Fifos
  Fifo write_fifo;  //Fifo uses to store the writes before going to the memory

  std::vector<Fifo> input_fifos;  //Fifos used to store the inputs before being fetched
  std::vector<Fifo> psum_fifos;   //Fifos used to store partial psums before being fetched
  //Fifo* read_fifo; //Fifo used to store the inputs before being fetched
  //Fifo* psums_fifo; //Fifo used to store partial psums before being fetched

  //Addresses
  address_t MK_address;
  address_t KN_address;
  address_t output_address;

  //Metadata addresses
  metadata_address_t MK_col_id;
  metadata_address_t MK_row_pointer;

  /* SST variables */
  uint64_t weight_dram_location;
  uint64_t input_dram_location;
  uint64_t output_dram_location;

  uint32_t data_width;
  uint32_t n_write_mshr;

  //Tile parameters
  std::size_t T_N_min;  //Minimum value of T_N
  std::size_t T_N;      //Actual value of T_N if adaptive tiling is used
  std::size_t T_K_max;  //This is the maximum value of tile of K
  std::size_t T_K;      //This is the actual value of tile of K
  std::size_t iter_N;
  std::size_t iter_K;  //This one will change for every value of V (vertex)
  std::size_t iter_M;

  //Current parameters
  std::size_t current_M;
  std::size_t current_N;
  std::size_t current_K_nnz;
  std::size_t K_nnz;

  //Counters to calculate SRC and DST
  std::size_t* sta_counters_table;  //Matrix of size rows*columns to figure out the dst of each sta value
  std::size_t* str_counters_table;  //Matrix of size rows*columns of the str matrix to calculate the source of each bit enabled.

  //Pointers
  std::size_t str_current_index;           //Streaming current index to calculate the next values to stream
  std::size_t sta_current_index_metadata;  //Stationary matrix current index (e.g., row in MK)
  std::size_t sta_current_index_matrix;    //Index to next element in the sparse matrix
  std::size_t sta_current_j_metadata;      //Index to current element in the same cluster. Used to manage folding
  std::size_t sta_last_j_metadata;         //Indext to last element in the same cluster. Used to manage folding
  std::size_t STA_base;
  //the boundaries of a certain fold is sta_current_j_metadata and sta_last_j_metadata

  //Signals
  bool configuration_done;      //Indicates whether the architecture has been configured to perform the delivering
  bool stationary_distributed;  //Indicates if the stationary values has been distributed for a certain iteration
  bool stationary_finished;     //Flag that indicates that all the stationary values have been delivered
  bool stream_finished;         //Flag that indicates that all the streaming values have been delivered
  bool execution_finished;      //Flag that indicates when the execution is over. This happens when all the output values have been calculated.
  bool sta_iter_completed;      //Indicates if the pending psums have been writen back
  bool STA_complete;

  bool metadata_loaded;  //Flag that indicates whether the metadata has been loaded
  bool layer_loaded;     //Flag that indicates whether the layer has been loaded.

  std::size_t current_output;
  std::size_t output_size;

  std::size_t current_output_iteration;
  std::size_t output_size_iteration;

  //For stats
  std::size_t n_ones_sta_matrix;
  std::size_t n_ones_str_matrix;
  std::vector<Connection*> write_port_connections;
  cycles_t local_cycle;
  SDMemoryStats sdmemoryStats;  //To track information

  //SST Memory hierarchy component structures and variables
  Memory<float>& mem;

  //Aux functions
  void receive();
  void send();
  void sendPackageToInputFifos(DataPackage* pck);

  std::vector<Connection*> getWritePortConnections() const { return this->write_port_connections; }

 public:
  SparseDenseSDMemory(stonne_id_t id, std::string name, Config stonne_cfg, Connection* write_connection, Memory<float>& mem);
  ~SparseDenseSDMemory();
  void setLayer(DNNLayer* dnn_layer, address_t KN_address, address_t MK_address, address_t output_address, Dataflow dataflow);

  void setReadConnections(std::vector<Connection*> read_connections);
  void setWriteConnections(std::vector<Connection*> write_port_connections);  //All the write connections must be set at a time
  void cycle();
  bool isExecutionFinished();

  void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer);  // Supported by this controller
  void setDenseSpatialData(std::size_t T_N, std::size_t T_K);

  void setReduceNetwork(ReduceNetwork* reduce_network) { this->reduce_network = reduce_network; }

  //Used to configure the MultiplierNetwork according to the controller
  void setMultiplierNetwork(MultiplierNetwork* multiplier_network) { this->multiplier_network = multiplier_network; }

  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);

  SDMemoryStats getStats() { return this->sdmemoryStats; }

  virtual void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) {}

  virtual void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, metadata_address_t KN_metadata_id,
                                       metadata_address_t KN_metadata_pointer) {}

  virtual void setTile(Tile* current_tile) {}
};

#endif  //SPARSESDMEMORY_H_
