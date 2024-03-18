//Created 19/02/2020

#ifndef __ACCUMULATOR__h
#define __ACCUMULATOR__h

#include <vector>
#include "comm/Connection.hpp"
#include "comm/DataPackage.hpp"
#include "common/Config.hpp"
#include "common/Fifo.hpp"
#include "common/Stats.hpp"
#include "common/Unit.hpp"
#include "common/types.hpp"

/*
*/

class Accumulator : public Unit {
 private:
  std::size_t input_ports;       // Input port
  std::size_t output_ports;      // output port
  std::size_t buffers_capacity;  //Buffers size in bytes
  std::size_t port_width;        //Bit width of each port

  std::size_t busID;    //CollectionBus connected to this ASwitch
  std::size_t inputID;  //Number of input of the Collection Bus busID connected to this AS.

  //Inputs fifos
  Fifo input_fifo;  // Array of packages that are received from the adders

  // Output Fifos
  Fifo output_fifo;  // Output fifo to the parent

  adderoperation_t operation_mode;  //Adder or comp

  std::size_t current_capacity;  // the capacity must not exceed the input bw of the connection
  Connection* inputConnection;   // This is the input left connection of the Adder
  Connection* outputConnection;  // This is the output connection of the adder

  cycles_t latency;  // Number of cycles to compute a sum. This is configurable since can vary depending on the implementation technology and frequency

  //Operation functions. This functions can be changed in order to perform different types of length operations
  DataPackage* perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right);  //Perform 2:1 sum

  cycles_t local_cycle;
  ASwitchStats aswitchStats;  //To track the behaviour of the FEASwitch

  //Extensions
  DataPackage* temporal_register;  //Temporal register to accumulate partial sums
  std::size_t n_psums;             //Number of psums before accumulation
  std::size_t current_psum;        //Current psum performed
  std::size_t n_accumulator;

  AccumulatorStats accumulatorStats;  //Object to track the behaviour of the Accumulator

  //Private functions
  void route();

 public:
  Accumulator(stonne_id_t id, std::string name, Config stonne_cfg, std::size_t n_accumulator);
  Accumulator(stonne_id_t id, std::string name, Config stonne_cfg, std::size_t n_accumulator, Connection* inputConnection, Connection* outputConnection);

  //Connection setters
  void setInputConnection(Connection* inputLeftConnection);  // Set the input left connection of the Adder
  void setOutputConnection(Connection* outputConnection);    // Set the output connection of the Adder
  void setNPSums(std::size_t n_psums);
  void resetSignals();

  // Getters
  std::size_t getNAcummulator() const { return this->n_accumulator; }

  std::size_t getOutputPorts() const { return this->output_ports; }  // Get the output ports

  // Functionality
  void send();     //Packages of data to be sent depending on routing.
  void receive();  //take data from connections

  void cycle();  //Computing a cycle. Based on routing the AS decides where the data goes.

  void printConfiguration(std::ofstream& out,
                          std::size_t indent);  //This function prints the configuration of FEASwitch such as  the operation mode, augmented link enabled, etc
  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);
};

#endif
