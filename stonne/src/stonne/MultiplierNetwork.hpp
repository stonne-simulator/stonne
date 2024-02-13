
#ifndef __MULTIPLIERNETWORK__H__
#define __MULTIPLIERNETWORK__H__

#include <assert.h>
#include <iostream>
#include "CompilerMSN.hpp"
#include "Connection.hpp"
#include "DNNLayer.hpp"
#include "DSwitch.hpp"
#include "MSwitch.hpp"
#include "Tile.hpp"
#include "Unit.hpp"

#include <map>

class MultiplierNetwork : public Unit {
 public:
  /*
       By the default the implementation of the MS just receives a single element, calculate a single psum and/or send a single input activation to the neighbour. This way, the parameters
       input_ports, output_ports and forwarding_ports will be set as the single data size. If this implementation change for future tests, this can be change easily bu mofifying these three parameters.
     */
  MultiplierNetwork(stonne_id_t id, std::string name) : Unit(id, name) {}

  virtual ~MultiplierNetwork() {}

  //set connections from the distribution network to the multiplier network
  virtual void setInputConnections(std::map<int, Connection*> input_connections) = 0;

  virtual void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections) = 0;

  //Set connections from the Multiplier Network to the Reduction Network
  virtual void setOutputConnections(std::map<int, Connection*> output_connections) = 0;

  virtual void cycle() = 0;

  virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding) = 0;

  virtual void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t ms_size) = 0;

  virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding, multiplierconfig_t multiplierconfig) = 0;

  virtual void resetSignals() = 0;

  virtual void printConfiguration(std::ofstream& out, std::size_t indent) = 0;

  virtual void printStats(std::ofstream& out, std::size_t indent) = 0;

  virtual void printEnergy(std::ofstream& out, std::size_t indent) = 0;
};
#endif
