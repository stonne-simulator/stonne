#ifndef __lookuptable__h
#define __lookuptable__h

#include <iostream>
#include "Config.hpp"
#include "Connection.hpp"
#include "Unit.hpp"

class LookupTable : Unit {
 private:
  Connection* inputConnection;   //From the ART
  Connection* outputConnection;  //Torwards the memory
  cycles_t latency;
  std::size_t port_width;

 public:
  LookupTable(stonne_id_t id, std::string name, Config stonne_cfg, Connection* inputConnection, Connection* outputConnection);
  void cycle();

  void printStats(std::ofstream& out, std::size_t indent) {}

  void printEnergy(std::ofstream& out, std::size_t indent) {}
};

#endif
