#ifndef __DSNETWORK__H__
#define __DSNETWORK__H__

#include <fstream>
#include <iostream>
#include <map>
#include "../../common/Config.hpp"
#include "../../common/Unit.hpp"
#include "../mn/MSNetwork.hpp"
#include "switch/DSwitch.hpp"

#define CONNECTIONS_PER_SWITCH 2
#define LEFT 0
#define RIGHT 1

class DSNetwork : public Unit {
 private:
  std::size_t m_msSize;  //Number of multipliers. i.e., the leaves of the network
  std::size_t m_portWidth;
  int m_nlevels;  //Number of levels of the DS without taking into account the MS level
  std::map<std::pair<int, int>, DSwitch*>
      m_dswitchtable;  //Map with the switches of the topology. The connection among them will be different depending on the topology used
  std::map<std::pair<int, int>, Connection*> m_connectiontable;  // Outputs connections of each level.
  Connection* p_inputConnection;                                 //Given by external

 public:
  DSNetwork(stonne_id_t id, std::string name, Config stonne_cfg, std::size_t ms_size,
            Connection* inputConnection);  //ms_size = ms_size of the group that contain this tree
  ~DSNetwork();

  int getNLevels() const { return this->m_nlevels; }

  int getMsSize() const { return this->m_msSize; }

  std::map<int, Connection*> getLastLevelConnections();

  void setInputConnection(Connection* inputConnection) {
    this->p_inputConnection = inputConnection;
  }  //This function set the inputConnection with the Prefetch buffer

  //Useful functions

  //Cycle function
  void cycle();
  unsigned long get_time_routing();
  void printStats(std::ofstream& out, std::size_t indent);  //Print the stats of the component
  void printEnergy(std::ofstream& out, std::size_t indent);
};

#endif
