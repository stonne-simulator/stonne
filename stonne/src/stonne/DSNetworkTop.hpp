// Created on 06/11/2019 by Francisco Munoz Martinez

#ifndef __DSNetworkTop__
#define __DSNetworkTop__

#include <iostream>
#include <vector>

#include <iostream>
#include <map>
#include "Connection.hpp"
#include "DSNetwork.hpp"
#include "DistributionNetwork.hpp"

//This class represents the whole DSNetwork that is composed by several DSNetworks trees. Basically, a DSNetwork has as many trees as input ports has the architecture to fetch input data.

class DSNetworkTop : public DistributionNetwork {
 private:
  std::size_t n_input_ports;
  std::size_t ms_size_per_port;  //Number of multipliers per each ds tree
  std::size_t port_width;

  std::vector<DSNetwork*> dsnetworks;    //one per port
  std::vector<Connection*> connections;  //one per port

 public:
  //The constructor creates n_input_ports dsnetworks with portWidth port_width
  DSNetworkTop(stonne_id_t id, std::string name, Config stonne_cfg);
  ~DSNetworkTop();
  void cycle();                                          //This just executes cycle over all the dsnetworks
  std::map<int, Connection*> getLastLevelConnections();  //Get last levels connections from all the astrees together. Useful to connect with mswitches later.
  std::vector<Connection*> getTopConnections();          // Get the top connections (i.e., the ones that connect the SDMemory ports with the DS subtrees)
  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);
};

#endif
