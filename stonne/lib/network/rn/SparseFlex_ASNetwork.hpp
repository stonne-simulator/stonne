#ifndef __SPARSEFLEX_ASNETWORK__H__
#define __SPARSEFLEX_ASNETWORK__H__

#include <fstream>
#include <iostream>
#include <map>
#include "../../common/Config.hpp"
#include "../../common/types.hpp"
#include "../../compiler/CompilerART.hpp"
#include "../../memory/AccumulationBuffer.hpp"
#include "../mn/MSNetwork.hpp"
#include "ReduceNetwork.hpp"
#include "switch/SparseFlex_ASwitch.hpp"

#define CONNECTIONS_PER_SWITCH 2
#define LEFT 0
#define RIGHT 1

class SparseFlex_ASNetwork : public ReduceNetwork {
 private:
  std::size_t port_width;  //Width in bits of each port
  std::size_t ms_size;     //Number of multipliers. i.e., the leaves of the network
  int nlevels;             //Number of levels of the AS without taking into account the MS level
  std::map<std::pair<int, int>, SparseFlex_ASwitch*>
      aswitchtable;  //Map with the switches of the topology. The connection among them will be different depending on the topology used

  //Copy of the pointers of the map aswitchtable used to generate the connections between the ART and the bus in the same way as the implementation in bluespec does.
  std::vector<SparseFlex_ASwitch*>
      single_switches;  //List of switches that are single reduction switches  (see blueSpec implementation. i.e., the do not have forwarding connections)
  std::vector<SparseFlex_ASwitch*>
      double_switches;  //List of double switches that are double reduction switches (i.e., in bluespec implementation these are sw that have fw links).

  std::map<std::pair<int, int>, Connection*> inputconnectiontable;       // input connections of each level.
  std::map<std::pair<int, int>, Connection*> forwardingconnectiontable;  // Forwarding connections of each level (intermedium links)
  std::vector<Connection*> accumulationbufferconnectiontable;            //Connections to the accumulation buffer if it is used
  Connection* outputConnection;                                          //Given by external
  AccumulationBuffer* accumulationBuffer;                                //Array of accumulators to perform the folding accumulation
  bool accumulation_buffer_enabled;
  Config stonne_cfg;

 public:
  SparseFlex_ASNetwork(stonne_id_t id, std::string name, Config stonne_cfg, Connection* output_connection);
  ~SparseFlex_ASNetwork();

  int getNLevels() const { return this->nlevels; }

  int getMsSize() const { return this->ms_size; }

  std::vector<SparseFlex_ASwitch*> getSingleSwitches() { return this->single_switches; }

  std::vector<SparseFlex_ASwitch*> getDoubleSwitches() { return this->double_switches; }

  void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections);  //Connect all the memory ports (busID, lineID) to its corresponding AS
  std::map<int, Connection*> getLastLevelConnections();

  void setOutputConnection(Connection* outputConnection) {
    this->outputConnection = outputConnection;
  }  //This function set the outputConnection with the Prefetch buffer

  void addersConfiguration(std::map<std::pair<int, int>, adderconfig_t> adder_configurations);
  void forwardingConfiguration(std::map<std::pair<int, int>, fl_t> fl_configurations);  //Configure the forwarding links. Enable the required ones.
  void childsLinksConfiguration(std::map<std::pair<int, int>, std::pair<bool, bool>> childs_configuration);
  void forwardingToMemoryConfiguration(std::map<std::pair<int, int>, bool> forwarding_to_memory_enabled);
  void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding);
  void configureSignalsSortTree(adderconfig_t sort_configuration);
  void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t ms_size);
  void resetSignals();

  //Cycle function
  void cycle();

  void printConfiguration(
      std::ofstream& out,
      std::size_t
          indent);  //This function prints the configuration of the SparseFlex_ASNetwork (i.e., SparseFlex_ASwitches configuration such as ADD_2_1, ADD_3_1, etc)
  void printStats(std::ofstream& out, std::size_t indent);  //This functions prints the statistics obtained during the execution.
  void printEnergy(std::ofstream& out, std::size_t indent);
};

#endif