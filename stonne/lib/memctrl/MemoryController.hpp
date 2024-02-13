#ifndef __MEMORYCONTROLLER__H__
#define __MEMORYCONTROLLER__H__

#include <assert.h>
#include <list>
#include "../comm/Connection.hpp"
#include "../comm/DataPackage.hpp"
#include "../common/Config.hpp"
#include "../common/Fifo.hpp"
#include "../common/Stats.hpp"
#include "../common/Unit.hpp"
#include "../common/dnn/DNNLayer.hpp"
#include "../common/dnn/Tile.hpp"
#include "../common/types.hpp"
#include "../network/mn/MultiplierNetwork.hpp"
#include "../network/rn/ReduceNetwork.hpp"

class MemoryController : Unit {
 public:
  MemoryController(stonne_id_t id, std::string name) : Unit(id, name) {}

  virtual ~MemoryController() {}

  virtual void setLayer(DNNLayer* dnn_layer, address_t input_address, address_t filter_address, address_t output_address, Dataflow dataflow) = 0;

  virtual void setTile(Tile* current_tile) = 0;

  virtual void setReadConnections(std::vector<Connection*> read_connections) = 0;

  virtual void setWriteConnections(std::vector<Connection*> write_port_connections) = 0;  //All the write connections must be set at a time

  virtual void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) = 0;

  //Used to configure the ReduceNetwork according to the controller if needed
  virtual void setReduceNetwork(ReduceNetwork* reduce_network) = 0;

  //Used to configure the MultiplierNetwork according to the controller if needed
  virtual void setMultiplierNetwork(MultiplierNetwork* multiplier_network) = 0;

  virtual void cycle() = 0;

  virtual bool isExecutionFinished() = 0;

  virtual void setDenseSpatialData(std::size_t T_N, std::size_t T_K) = 0;

  virtual void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer) = 0;

  virtual void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, metadata_address_t KN_metadata_id,
                                       metadata_address_t KN_metadata_pointer) = 0;

  virtual void printStats(std::ofstream& out, std::size_t indent) = 0;

  virtual void printEnergy(std::ofstream& out, std::size_t indent) = 0;
};

#endif  //SDMEMORY_H_
