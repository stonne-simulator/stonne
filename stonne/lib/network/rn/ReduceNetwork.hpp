#ifndef __REDUCENETWORK__H__
#define __REDUCENETWORK__H__

#include <fstream>
#include <iostream>
#include <map>
#include "../../comm/Connection.hpp"
#include "../../common/Config.hpp"
#include "../../common/Unit.hpp"
#include "../../common/dnn/DNNLayer.hpp"
#include "../../common/dnn/Tile.hpp"
#include "../../common/types.hpp"
#include "../../compiler/CompilerComponent.hpp"

class ReduceNetwork : public Unit {

 public:
  ReduceNetwork(stonne_id_t id, std::string name) : Unit(id, name) {}

  virtual ~ReduceNetwork() {}

  virtual void setMemoryConnections(
      std::vector<std::vector<Connection*>> memoryConnections) = 0;  //Connect all the memory ports from buses (busID, lineID) to its corresponding switches

  virtual std::map<int, Connection*> getLastLevelConnections() = 0;

  virtual void setOutputConnection(Connection* outputConnection) = 0;  //This function set the outputConnection with the Prefetch buffer

  virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding) = 0;

  virtual void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t ms_size) = 0;

  virtual void configureSignalsSortTree(adderconfig_t sort_configuration) = 0;

  virtual void resetSignals() = 0;

  //Cycle function
  virtual void cycle() {}

  virtual void printConfiguration(
      std::ofstream& out,
      std::size_t indent) = 0;  //This function prints the configuration of the ASNetwork (i.e., ASwitches configuration such as ADD_2_1, ADD_3_1, etc)

  virtual void printStats(std::ofstream& out, std::size_t indent) = 0;  //This functions prints the statistics obtained during the execution.

  virtual void printEnergy(std::ofstream& out, std::size_t indent) = 0;
};

#endif
