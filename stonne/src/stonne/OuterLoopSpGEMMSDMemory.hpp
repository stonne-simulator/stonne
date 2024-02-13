#ifndef __OUTERSPGEMMSDMEMORY__H__
#define __OUTERSPGEMMSDMEMORY__H__

#include <list>
#include "Config.hpp"
#include "Connection.hpp"
#include "DNNLayer.hpp"
#include "DataPackage.hpp"
#include "Fifo.hpp"
#include "Memory.hpp"
#include "MemoryController.hpp"
#include "MultiplierNetwork.hpp"
#include "ReduceNetwork.hpp"
#include "Stats.hpp"
#include "Tile.hpp"
#include "Unit.hpp"
#include "types.hpp"

class OuterLoopSpGEMMSDMemory : public MemoryController {
 private:
  DNNLayer* dnn_layer;                    // Layer loaded in the accelerator
  ReduceNetwork* reduce_network;          //Reduce network used to be reconfigured
  MultiplierNetwork* multiplier_network;  //Multiplier network used to be reconfigured

  std::vector<std::queue<DataPackage*>>* intermediate_memory;
  std::vector<std::queue<DataPackage*>> swap_memory;              //To be used during the several iterations
  std::vector<std::queue<DataPackage*>>* pointer_current_memory;  //This is to exchange between a particular row in intermediate_memory and swap_memory
  std::vector<std::queue<DataPackage*>>* pointer_next_memory;

  std::size_t M;
  std::size_t N;
  std::size_t K;  //Number of columns MK matrix and rows KN matrix. Extracted from dnn_layer->get_C();

  Connection* write_connection;
  SparsityControllerState current_state;   //Stage to control what to do according to the state
  std::vector<SparseVN> configurationVNs;  //A set of each VN size mapped onto the architecture.
  std::vector<int> vnat_table;
  std::vector<int> ms_group;
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
  Fifo* write_fifo;  //Fifo uses to store the writes before going to the memory

  std::vector<Fifo*> input_fifos;  //Fifos used to store the inputs before being fetched
  std::vector<Fifo*> psum_fifos;   //Fifos used to store partial psums before being fetched
  //Fifo* read_fifo; //Fifo used to store the inputs before being fetched
  //Fifo* psums_fifo; //Fifo used to store partial psums before being fetched

  //Addresses
  address_t MK_address;
  address_t KN_address;
  address_t output_address;

  //Metadata addresses
  metadata_address_t MK_row_id;
  metadata_address_t MK_col_pointer;  //Actually this is col pointer, but the functionality is the same.
  metadata_address_t KN_col_id;
  metadata_address_t KN_row_pointer;

  //SST Memory hierarchy component structures and variables
  Memory<float>& mem;

  //Current pointers
  std::size_t current_MK;
  std::size_t current_MK_col_pointer;
  std::size_t current_MK_row_id;
  std::size_t current_KN;
  std::size_t current_KN_row_pointer;
  std::size_t current_KN_col_id;

  /* SST variables */
  uint64_t weight_dram_location;
  uint64_t input_dram_location;
  uint64_t output_dram_location;

  uint32_t data_width;
  uint32_t n_write_mshr;

  //Aux parameters
  std::size_t MK_number_nnz;
  std::size_t multipliers_used;
  std::size_t n_str_data_sent;
  std::size_t n_str_data_received;

  //Signals
  bool configuration_done;      //Indicates whether the architecture has been configured to perform the delivering
  bool stationary_distributed;  //Indicates if the stationary values has been distributed for a certain iteration
  bool stationary_finished;     //Flag that indicates that all the stationary values have been delivered
  bool stream_finished;         //Flag that indicates that all the streaming values have been delivered
  bool execution_finished;      //Flag that indicates when the execution is over. This happens when all the output values have been calculated.
  bool sta_iter_completed;      //Indicates if the pending psums have been writen back
  bool last_sta_iteration_completed;
  bool STA_complete;
  bool STR_complete;
  bool multiplication_phase_finished;
  bool sort_down_last_iteration_finished;
  bool sort_down_iteration_finished;
  bool sort_up_iteration_finished;
  bool sort_up_received_first_value;
  bool sort_up_exception_row_empty;

  bool metadata_loaded;  //Flag that indicates whether the metadata has been loaded
  bool layer_loaded;     //Flag that indicates whether the layer has been loaded.

  std::size_t current_output;
  std::size_t output_size;

  std::size_t current_output_iteration;
  std::size_t output_size_iteration;

  //SORTING TREE CONTROL
  std::size_t sort_col_id;
  std::size_t sort_row_id;
  std::size_t sort_sub_block_id;
  std::size_t sort_num_blocks;
  bool swap_memory_enabled;
  //For stats
  std::size_t n_ones_sta_matrix;
  std::size_t n_ones_str_matrix;
  std::vector<Connection*> write_port_connections;
  cycles_t local_cycle;
  SDMemoryStats sdmemoryStats;  //To track information

  Tile* tile;  //Not really used in sparseflex

  //Variable to manage the number of sorting iterations
  int sorting_iterations;
  int current_sorting_iteration;
  int n_values_stored;

  //Aux functions
  void receive();
  void send();
  bool doLoad(uint64_t addr, DataPackage* data_package);
  bool doStore(uint64_t addr, DataPackage* data_package);
  void sendPackageToInputFifos(DataPackage* pck);

  std::vector<Connection*> getWritePortConnections() const { return this->write_port_connections; }

 public:
  OuterLoopSpGEMMSDMemory(stonne_id_t id, std::string name, Config stonne_cfg, Connection* write_connection, Memory<float>& mem);
  ~OuterLoopSpGEMMSDMemory();
  void setLayer(DNNLayer* dnn_layer, address_t KN_address, address_t MK_address, address_t output_address, Dataflow dataflow);

  void setReadConnections(std::vector<Connection*> read_connections);
  void setWriteConnections(std::vector<Connection*> write_port_connections);  //All the write connections must be set at a time
  void cycle();
  bool isExecutionFinished();

  void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, metadata_address_t KN_metadata_id,
                               metadata_address_t KN_metadata_pointer);

  void setReduceNetwork(ReduceNetwork* reduce_network) { this->reduce_network = reduce_network; }

  //Used to configure the MultiplierNetwork according to the controller
  void setMultiplierNetwork(MultiplierNetwork* multiplier_network) { this->multiplier_network = multiplier_network; }

  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);

  SDMemoryStats getStats() { return this->sdmemoryStats; }

  virtual void setTile(Tile* current_tile) {}

  void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) {}

  void setDenseSpatialData(std::size_t T_N, std::size_t T_K) {}

  void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer) {}
};

#endif  //SPARSESDMEMORY_H_
