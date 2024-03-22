#ifndef __SDMEMORY__H__
#define __SDMEMORY__H__

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

//This class contains for each VN the next address to write
class VNAT_Register {
 public:
  std::size_t VN;         //VN Saved
  std::size_t base_addr;  //Base addr of this VN (i.e., the first element to compute).
  std::size_t addr;       //Offset
  std::size_t current_N;
  std::size_t current_G;
  std::size_t current_K;
  std::size_t current_X;
  std::size_t current_Y;
  std::size_t current_R;
  std::size_t current_S;
  std::size_t current_C;
  //To calculate next output_address
  std::size_t iter_N;
  std::size_t iter_G;
  std::size_t iter_K;
  std::size_t iter_X;
  std::size_t iter_Y;
  std::size_t iter_R;
  std::size_t iter_S;
  std::size_t iter_C;
  std::size_t n_psums;  //psums per window
  std::size_t current_psum;
  DNNLayer* dnn_layer;
  Tile* current_tile;
  bool finished;

  VNAT_Register(std::size_t VN, std::size_t addr, std::size_t N, std::size_t G, std::size_t K, std::size_t X, std::size_t Y, std::size_t iter_N,
                std::size_t iter_G, std::size_t iter_K, std::size_t iter_X, std::size_t iter_Y, std::size_t iter_R, std::size_t iter_S, std::size_t iter_C,
                DNNLayer* dnn_layer, Tile* current_tile);
  void update();  //Update variables to the next cycle
  std::size_t get_address();
};

class SDMemory : public MemoryController {
 private:
  DNNLayer* dnn_layer;  // Layer loaded in the accelerator
  Tile* current_tile;   // Layer loaded in the tile
  ReduceNetwork*
    reduce_network;  //This is not used in this controller as the configuration is performed in STONNEModel when the tile is loaded, and this is needed just once
  MultiplierNetwork* multiplier_network;  //Idem as reduce_network
  Connection* write_connection;
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
  address_t filter_address;
  address_t input_address;
  address_t output_address;

  uint64_t weight_dram_location;
  uint64_t input_dram_location;
  uint64_t output_dram_location;

  uint32_t data_width;
  uint32_t n_write_mshr;

  //Signals
  bool weights_distributed;  //Indicates if the weights have been distributed for a certain iteration
  bool fw_link_enabled;      //Indicates if the fw link is enabled in this cycle and therefore the number of bw used per cycle is less
  bool weights_finished;     //Flag that indicates that all the weights have been delivered
  bool input_finished;       //Flag that indicates that all the inputs have been delivered
  bool tile_loaded;          //SPecify if the tile is loaded
  bool execution_finished;   //Flag that indicates when the execution is over. This happens when all the opixels have been calculated.

  //Variables to track the progress of the execution
  std::size_t iter_R;
  std::size_t iter_S;
  std::size_t iter_C;
  std::size_t iter_G;
  std::size_t iter_N;
  std::size_t iter_K;
  std::size_t iter_X;
  std::size_t iter_Y;

  std::size_t current_R;
  std::size_t current_S;
  std::size_t current_C;
  std::size_t current_G;
  std::size_t current_N;
  std::size_t current_K;
  std::size_t current_X;
  std::size_t current_Y;

  //Variable to track the number of opixels calculated
  std::size_t current_output_pixel;      //This variable has the count for the current number of output pixels calculated
  std::size_t output_pixels_to_compute;  //This variable has the number of output pixels that the simulator must calculate before finishing the execution
  std::size_t output_psums_per_channel;

  //Variables to make the calculation easier
  std::size_t channel_filter_size;
  std::size_t row_filter_size;
  std::size_t filter_size;
  std::size_t channel_input_size;
  std::size_t row_input_size;
  std::size_t input_size;
  std::size_t channel_output_size;
  std::size_t row_output_size;
  std::size_t output_size;
  std::size_t group_size;

  std::list<DataPackage*> packages_created;  // Vector used to track the packages and delete them at the end of the execution
  std::vector<Connection*> write_port_connections;
  VNAT_Register** VNAT;  //VNAT with as many registers as VN configured in the accelerator
  cycles_t local_cycle;
  SDMemoryStats sdmemoryStats;  //To track information

  //SST Memory hierarchy component structures and variables
  Memory<float>& mem;

  //Aux functions
  void receive();
  void sendPackageToInputFifos(DataPackage* pck);
  void send();

  std::vector<Connection*> getWritePortConnections() const { return this->write_port_connections; }

 public:
  SDMemory(stonne_id_t id, std::string name, Config stonne_cfg, Connection* write_connection, Memory<float>& mem);
  ~SDMemory();
  void setLayer(DNNLayer* dnn_layer, address_t input_address, address_t filter_address, address_t output_address, Dataflow dataflow);
  void setTile(Tile* current_tile);
  void setReadConnections(std::vector<Connection*> read_connections);
  void setWriteConnections(std::vector<Connection*> write_port_connections);  //All the write connections must be set at a time

  void setReduceNetwork(ReduceNetwork* reduce_network) { this->reduce_network = reduce_network; }

  //Used to configure the MultiplierNetwork according to the controller if needed
  void setMultiplierNetwork(MultiplierNetwork* multiplier_network) { this->multiplier_network = multiplier_network; }

  void cycle();
  bool isExecutionFinished();

  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);

  void setDenseSpatialData(std::size_t T_N, std::size_t T_K) {}

  void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer) {}

  void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, metadata_address_t KN_metadata_id,
                               metadata_address_t KN_metadata_pointer) {}

  void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) {}
};

#endif  //SDMEMORY_H_
