//Created 13/06/2019

#ifndef __DSwitch__h
#define __DSwitch__h

#include "types.h"
#include "DataPackage.h"
#include "Connection.h"
#include "Unit.h"
#include "Config.h"
#include <vector>
#include "Stats.h"
/*
*/

class DSwitch : public Unit{
private:
    unsigned int level; //Level where the switch is set in the tree
    unsigned int num_in_level;
    unsigned int num_ms; //These three parameters are for routing. In hardware it is not neccesary since it is used a bit vector
    bool pending_data;   // Indicates if data exists
    unsigned int input_ports;           // Number of input ports in the DSwitch
    unsigned int output_ports;     //Number of output ports in the DSwitch
    unsigned int port_width;
    std::vector<DataPackage*> data;   // Array of packages that are send/receive in  a certain cycle. The number of packages depends on the bw of the connection. Even though the switches are bufferless, this is prepared for future implementations. In the first case in which the switches are bufferless,
    //in every cycle the elements will be writen in the array and read right after. 
    unsigned int current_capacity; // the capacity must not exceed the input bw of the connection
    Connection* leftConnection;   // This is the left connection of the switch
    Connection* rightConnection;  // This is the right connection of the switch
    Connection* inputConnection;
    latency_t latency;
    
    ///Aux functions
    void route_packages(); // Used to send the packages depending on the type (BROADCAST, UNICAST or MULTICAST)

    //DEBUG PARAMETERS
    unsigned long time_routing;
    //unsigned long time_receive;
    //unsigned long time_send;
    DSwitchStats dswitchStats; //contains the counters to track the behaviour of the DSwitch


public:
    //Since input_ports and output_ports depends on the level of the tree, this cannot be a configuring parameter and has to be set at the moment of creating the network
    DSwitch(id_t id, std::string name, unsigned int level, unsigned int num_in_level, Config stonne_cfg, unsigned int ms_size); //Output bandwidth is the bw per branch
    DSwitch(id_t id, std::string name, unsigned int level, unsigned int num_in_level, Config stonne_cfg, unsigned int ms_size, Connection* leftConnection, Connection* rightConnection, Connection* inputConnection);
    void setLeftConnection(Connection* leftConnection); //Set the left connection of the switch
    void setRightConnection(Connection* rightConnection); //Set the right connection of the switch
    void setInputConnection(Connection* inputConnection); //Set the input connection of the switch
    const unsigned int getInputPorts() const {return this->input_ports;} //Get the input ports
    const unsigned int getOutputPorts() const {return this->output_ports;} //get the output ports
    void send(std::vector<DataPackage*> data, Connection* connection); //Packages of data to be send depending on routing. 
    void receive();  //Receive a list of  packages from the Inputconnection and save it in this->data
    void cycle();
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    unsigned long get_time_routing() const {return this->time_routing;}
	

};

#endif

