#ifndef __ACCUMULATIONBUFFER__H__
#define __ACCUMULATIONBUFFER__H__

#include <fstream>
#include <iostream>
#include <map>
#include "Accumulator.hpp"
#include "Config.hpp"
#include "DNNLayer.hpp"
#include "Tile.hpp"
#include "types.hpp"

class AccumulationBuffer : public Unit {
 private:
  std::size_t port_width;                        //Width in bits of each port
  std::size_t n_accumulators;                    //Number of accumulator array
  std::map<int, Accumulator*> accumulatortable;  //Map with the accumulators

  std::map<int, Connection*> inputconnectiontable;   // input connections
  std::map<int, Connection*> outputconnectiontable;  // Output connections
 public:
  AccumulationBuffer(stonne_id_t id, std::string name, Config stonne_cfg, std::size_t n_accumulators);
  ~AccumulationBuffer();
  void setMemoryConnections(std::vector<Connection*> memoryConnections);
  void setInputConnections(std::vector<Connection*> inputConnections);
  void resetSignals();
  void NPSumsConfiguration(std::size_t n_psums);

  void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding);
  //Cycle function
  void cycle();

  void printConfiguration(std::ofstream& out, std::size_t indent);
  void printStats(std::ofstream& out, std::size_t indent);   //This functions prints the stats
  void printEnergy(std::ofstream& out, std::size_t indent);  //Print the counters
};

#endif
