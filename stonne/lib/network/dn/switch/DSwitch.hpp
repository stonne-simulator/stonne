//Created 13/06/2019

#ifndef __DSwitch__h
#define __DSwitch__h

#include <vector>
#include "../../../comm/Connection.hpp"
#include "../../../comm/DataPackage.hpp"
#include "../../../common/Config.hpp"
#include "../../../common/Stats.hpp"
#include "../../../common/Unit.hpp"
#include "../../../common/types.hpp"

/*
*/

class DSwitch : public Unit {
 private:
  std::size_t level;  //Level where the switch is set in the tree
  std::size_t num_in_level;
  std::size_t num_ms;        //These three parameters are for routing. In hardware it is not neccesary since it is used a bit vector
  bool pending_data;         // Indicates if data exists
  std::size_t input_ports;   // Number of input ports in the DSwitch
  std::size_t output_ports;  //Number of output ports in the DSwitch
  std::size_t port_width;
  std::vector<DataPackage*>
      data;  // Array of packages that are send/receive in  a certain cycle. The number of packages depends on the bw of the connection. Even though the switches are bufferless, this is prepared for future implementations. In the first case in which the switches are bufferless,
  //in every cycle the elements will be writen in the array and read right after.
  std::size_t current_capacity;  // the capacity must not exceed the input bw of the connection
  Connection* leftConnection;    // This is the left connection of the switch
  Connection* rightConnection;   // This is the right connection of the switch
  Connection* inputConnection;
  latency_t latency;

  ///Aux functions
  void route_packages();  // Used to send the packages depending on the type (BROADCAST, UNICAST or MULTICAST)

  //DEBUG PARAMETERS
  unsigned long time_routing;
  //unsigned long time_receive;
  //unsigned long time_send;
  DSwitchStats dswitchStats;  //contains the counters to track the behaviour of the DSwitch

 public:
  //Since input_ports and output_ports depends on the level of the tree, this cannot be a configuring parameter and has to be set at the moment of creating the network
  DSwitch(stonne_id_t id, std::string name, std::size_t level, std::size_t num_in_level, Config stonne_cfg,
          std::size_t ms_size);  //Output bandwidth is the bw per branch
  DSwitch(stonne_id_t id, std::string name, std::size_t level, std::size_t num_in_level, Config stonne_cfg, std::size_t ms_size, Connection* leftConnection,
          Connection* rightConnection, Connection* inputConnection);
  void setLeftConnection(Connection* leftConnection);    //Set the left connection of the switch
  void setRightConnection(Connection* rightConnection);  //Set the right connection of the switch
  void setInputConnection(Connection* inputConnection);  //Set the input connection of the switch

  std::size_t getInputPorts() const { return this->input_ports; }  //Get the input ports

  std::size_t getOutputPorts() const { return this->output_ports; }  //get the output ports

  void send(std::vector<DataPackage*> data, Connection* connection);  //Packages of data to be send depending on routing.
  void receive();                                                     //Receive a list of  packages from the Inputconnection and save it in this->data
  void cycle();
  void printStats(std::ofstream& out, std::size_t indent);
  void printEnergy(std::ofstream& out, std::size_t indent);

  unsigned long get_time_routing() const { return this->time_routing; }
};

#endif
