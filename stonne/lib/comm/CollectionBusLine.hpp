// Created the 4th of november of 2019 by Francisco Munoz Martinez

#ifndef __CollectionBusLine__h__
#define __CollectionBusLine__h__

#include <iostream>
#include "../common/Fifo.hpp"
#include "../common/Stats.hpp"
#include "../common/Unit.hpp"
#include "Connection.hpp"

class CollectionBusLine : public Unit {

 private:
  std::size_t input_ports;                     //Number of input connections that correspond with input_connections.size() and input_fifos.size()
  std::vector<Connection*> input_connections;  //Every input connection for this bus line
  std::vector<Fifo*> input_fifos;              //Every fifo corresponds with an inputConnection for this busLine
  Connection* output_port;                     //Output connection with memory
  std::size_t next_input_selected;             //Using RR policy
  std::size_t busID;                           //Output port ID of this line

  void receive();
  CollectionBusLineStats collectionbuslineStats;  //To track information

 public:
  //Getters useful to make the connections with the ART switches and the memory
  std::vector<Connection*> getInputConnections() { return this->input_connections; }

  Connection* getOutputPort() { return this->output_port; }

  Connection* getInputPort(std::size_t inputID);

  //Creates the input_connections, the input_fifos and the output_port
  CollectionBusLine(stonne_id_t id, std::string name, std::size_t busID, std::size_t input_ports_bus_line, std::size_t connection_width, std::size_t fifo_size);
  ~CollectionBusLine();  //Destroy connection, fifos, and output connection
  void cycle();          //Select one input and send it trough the output

  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);
};

#endif
