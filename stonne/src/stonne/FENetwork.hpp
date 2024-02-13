#ifndef __FENETWORK__H__
#define __FENETWORK__H__

#include <fstream>
#include <iostream>
#include <map>
#include "CompilerFEN.hpp"
#include "Config.hpp"
#include "FEASwitch.hpp"
#include "MSNetwork.hpp"
#include "ReduceNetwork.hpp"
#include "types.hpp"

#define CONNECTIONS_PER_SWITCH 2
#define LEFT 0
#define RIGHT 1

class FENetwork : public ReduceNetwork {
 private:
  std::size_t m_portWidth;  //Width in bits of each port
  std::size_t m_msSize;     //Number of multipliers. i.e., the leaves of the network
  int m_nlevels;            //Number of levels of the AS without taking into account the MS level
  std::map<std::pair<int, int>, FEASwitch*>
      m_aswitchtable;  //Map with the switches of the topology. The connection among them will be different depending on the topology used

  //Copy of the pointers of the map aswitchtable used to generate the connections between the ART and the bus in the same way as the implementation in bluespec does.
  std::vector<FEASwitch*>
      m_singleSwitches;  //List of switches that are single reduction switches  (see blueSpec implementation. i.e., the do not have forwarding connections)
  std::vector<FEASwitch*>
      m_doubleSwitches;  //List of double switches that are double reduction switches (i.e., in bluespec implementation these are sw that have fw links).

  std::map<std::pair<int, int>, Connection*> m_inputconnectiontable;       // input connections of each level.
  std::map<std::pair<int, int>, Connection*> m_forwardingconnectiontable;  // Forwarding connections of each level (intermedium links)
  std::map<std::pair<int, int>, Connection*> m_foldingconnectiontable;     //Forwarding connections between each node and its folding manager
  Connection* p_outputConnection;                                          //Given by external

 public:
  FENetwork(stonne_id_t id, std::string name, Config stonne_cfg, Connection* output_connection);
  ~FENetwork();

  int getNLevels() const { return this->m_nlevels; }

  int getMsSize() const { return this->m_msSize; }

  std::vector<FEASwitch*> getSingleSwitches() { return this->m_singleSwitches; }

  std::vector<FEASwitch*> getDoubleSwitches() { return this->m_doubleSwitches; }

  void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections);  //Connect all the memory ports (busID, lineID) to its corresponding AS
  std::map<int, Connection*> getLastLevelConnections();

  void setOutputConnection(Connection* outputConnection) {
    this->p_outputConnection = outputConnection;
  }  //This function set the outputConnection with the Prefetch buffer

  void addersConfiguration(std::map<std::pair<int, int>, adderconfig_t> adder_configurations);
  void forwardingConfiguration(std::map<std::pair<int, int>, fl_t> fl_configurations);  //Configure the forwarding links. Enable the required ones.
  void childsLinksConfiguration(std::map<std::pair<int, int>, std::pair<bool, bool>> childs_configuration);
  void forwardingToMemoryConfiguration(std::map<std::pair<int, int>, bool> forwarding_to_memory_enabled);
  void forwardingToFoldNodeConfiguration(std::map<std::pair<int, int>, bool> forwarding_to_fold_node_enabled);
  void NPSumsConfiguration(std::size_t n_psums);

  void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding);

  //Cycle function
  void cycle();

  void printConfiguration(
      std::ofstream& out,
      std::size_t indent);  //This function prints the configuration of the ASNetwork (i.e., FEASwitches configuration such as ADD_2_1, ADD_3_1, etc)
  void printStats(std::ofstream& out, std::size_t indent);  //This functions prints the statistics obtained during the execution.
  void printEnergy(std::ofstream& out, std::size_t indent);

  void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t ms_size) {}

  void configureSignalsSortTree(adderconfig_t sort_configuration) {}

  void resetSignals() {}
};

#endif
