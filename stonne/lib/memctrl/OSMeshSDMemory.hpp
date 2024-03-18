#ifndef __OSMESHSDMEMORY__H__
#define __OSMESHSDMEMORY__H__

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
#include "network/mn/MultiplierNetwork.hpp"
#include "network/rn/ReduceNetwork.hpp"

class OSMeshSDMemory : public MemoryController {
 private:
  DNNLayer* dnn_layer;                    // Layer loaded in the accelerator
  ReduceNetwork* reduce_network;          //Reduce network used to be reconfigured
  MultiplierNetwork* multiplier_network;  //Multiplier network used to be reconfigured

  std::size_t M;
  std::size_t N;

  std::size_t K;  //Number of columns MK matrix and rows KN matrix. Extracted from dnn_layer->get_C();

  std::size_t OUT_DIST_VN;            //To calculate the output memory address
  std::size_t OUT_DIST_VN_ITERATION;  //To calculate the memory address
  Connection* write_connection;
  OSMeshControllerState current_state;     //Stage to control what to do according to the state
  std::vector<SparseVN> configurationVNs;  //A set of each VN size mapped onto the architecture.
  std::vector<std::size_t> vnat_table;     //Every element is a VN, indicating the column that is calculating
  //Connection* read_connection;
  std::vector<Connection*> read_connections;  //Input port connections. There are as many connections as n_read_ports are specified.

  //Input parameters
  std::size_t ms_rows;
  std::size_t ms_cols;
  std::size_t n_read_ports;
  std::size_t n_write_ports;
  std::size_t write_buffer_capacity;
  std::size_t port_width;

  std::size_t rows_used;
  std::size_t cols_used;

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

  //Tile parameters
  std::size_t T_N;  //Actual value of T_N if adaptive tiling is used
  std::size_t T_K;  //This is the actual value of tile of K. This is just 1 in this case
  std::size_t T_M;
  std::size_t iter_N;
  std::size_t iter_K;
  std::size_t iter_M;

  //Current parameters
  std::size_t current_M;
  std::size_t current_N;
  std::size_t current_K;

  //Signals
  bool configuration_done;  //Indicates whether the architecture has been configured to perform the delivering
  bool execution_finished;  //Flag that indicates when the execution is over. This happens when all the output values have been calculated.
  bool iteration_completed;

  bool metadata_loaded;  //Flag that indicates whether the metadata has been loaded
  bool layer_loaded;     //Flag that indicates whether the layer has been loaded.

  std::size_t current_output;
  std::size_t output_size;

  std::size_t current_output_iteration;
  std::size_t n_iterations_completed;
  std::size_t output_size_iteration;

  //For stats
  std::vector<Connection*> write_port_connections;
  cycles_t local_cycle;
  SDMemoryStats sdmemoryStats;  //To track information

  //Aux functions
  void receive();
  void send();
  void sendPackageToInputFifos(DataPackage* pck);

  std::vector<Connection*> getWritePortConnections() const { return this->write_port_connections; }

 public:
  OSMeshSDMemory(stonne_id_t id, std::string name, Config stonne_cfg, Connection* write_connection);
  void setLayer(DNNLayer* dnn_layer, address_t KN_address, address_t MK_address, address_t output_address, Dataflow dataflow);
  void setTile(Tile* current_tile);
  void setReadConnections(std::vector<Connection*> read_connections);
  void setWriteConnections(std::vector<Connection*> write_port_connections);  //All the write connections must be set at a time
  void cycle();
  bool isExecutionFinished();

  void setReduceNetwork(ReduceNetwork* reduce_network) { this->reduce_network = reduce_network; }

  //Used to configure the MultiplierNetwork according to the controller
  void setMultiplierNetwork(MultiplierNetwork* multiplier_network) { this->multiplier_network = multiplier_network; }

  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);

  void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) {}

  void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, metadata_address_t KN_metadata_id,
                               metadata_address_t KN_metadata_pointer) {}

  void setDenseSpatialData(std::size_t T_N, std::size_t T_K) {}

  void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer) {}
};

#endif
