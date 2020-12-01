// Created the 4th of november of 2019 by Francisco Munoz Martinez

#ifndef __CollectionBusLine__h__
#define __CollectionBusLine__h__

#include "Fifo.h"
#include "Connection.h"
#include <iostream>
#include "Unit.h"
#include "Stats.h"

class CollectionBusLine : public Unit {

private: 
    unsigned int input_ports;  //Number of input connections that correspond with input_connections.size() and input_fifos.size()
    std::vector<Connection*> input_connections; //Every input connection for this bus line
    std::vector<Fifo*> input_fifos; //Every fifo corresponds with an inputConnection for this busLine
    Connection* output_port;      //Output connection with memory
    unsigned int next_input_selected; //Using RR policy
    unsigned int busID;  //Output port ID of this line

    void receive();
    CollectionBusLineStats collectionbuslineStats; //To track information

public: 
    //Getters useful to make the connections with the ART switches and the memory
    std::vector<Connection*> getInputConnections() {return this->input_connections;}
    Connection* getOutputPort() {return this->output_port;}
    Connection* getInputPort(unsigned int inputID);
    
    //Creates the input_connections, the input_fifos and the output_port
    CollectionBusLine(id_t id, std::string name, unsigned int busID, unsigned int input_ports_bus_line, unsigned int connection_width, unsigned int fifo_size);
    ~CollectionBusLine(); //Destroy connection, fifos, and output connection
    void cycle(); //Select one input and send it trough the output

    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
   
    

};





#endif
