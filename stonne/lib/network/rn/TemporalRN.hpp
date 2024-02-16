
#ifndef __TEMPORALREDUCTIONNETWORK__H__
#define __TEMPORALREDUCTIONNETWORK__H__

#include <fstream>
#include <iostream>
#include <map>
#include "ReduceNetwork.hpp"
#include "common/Config.hpp"
#include "common/types.hpp"
#include "compiler/CompilerART.hpp"
#include "memory/AccumulationBuffer.hpp"
#include "network/mn/MSNetwork.hpp"
#include "switch/ASwitch.hpp"

class TemporalRN : public ReduceNetwork {
 private:
  std::size_t port_width;                          //Width in bits of each port
  std::vector<Connection*> inputconnectiontable;   //Connections to the accumulation buffer
  std::vector<Connection*> outputconnectiontable;  //Connection to the collection bus
  Connection* outputConnection;                    //Given by external
  AccumulationBuffer* accumulationBuffer;          //Array of accumulators to perform the folding accumulation
  std::size_t accumulation_buffer_size;            //Number of accumulation elements in the RN
  Config stonne_cfg;

 public:
  TemporalRN(stonne_id_t id, std::string name, Config stonne_cfg, Connection* output_connection);
  ~TemporalRN();
  void setMemoryConnections(
      std::vector<std::vector<Connection*>> memoryConnections);  //Connect all the memory ports (busID, lineID) to its corresponding accumulator
  std::map<int, Connection*> getLastLevelConnections();

  void setOutputConnection(Connection* outputConnection) {
    this->outputConnection = outputConnection;
  }  //This function set the outputConnection with the Prefetch buffer

  void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding);
  void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t ms_size);
  void resetSignals();

  //Cycle function
  void cycle();

  void printConfiguration(std::ofstream& out, std::size_t indent);
  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);

  virtual void configureSignalsSortTree(adderconfig_t sort_configuration) {}
};

#endif
