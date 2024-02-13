#ifndef _BUS_CPP
#define _BUS_CPP

#include <iostream>
#include <vector>
#include "CollectionBusLine.hpp"
#include "Config.hpp"
#include "Connection.hpp"
#include "Unit.hpp"

class Bus : public Unit {

 private:
  std::size_t n_bus_lines;  //Number of outputs from the bus
  std::size_t input_ports_bus_line;
  std::size_t connection_width;
  std::size_t fifo_size;
  std::vector<CollectionBusLine*> collection_bus_lines;

 public:
  Bus(stonne_id_t id, std::string name, Config stonne_cfg);
  ~Bus();

  std::size_t getNBusLines() { return this->n_bus_lines; }

  std::size_t getInputPortsBusLine() { return this->input_ports_bus_line; }

  std::vector<std::vector<Connection*>> getInputConnections();
  std::vector<Connection*> getOutputConnections();                                    //Get the output connections of all the lines
  Connection* getInputConnectionFromBusLine(std::size_t busID, std::size_t inputID);  //Get a specific inpur from a specific bus line

  void cycle();  //Get the inputs and send as many as posssible to the outputs

  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);
};

#endif
