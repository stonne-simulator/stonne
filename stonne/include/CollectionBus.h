// Created by Francisco Munoz on 28/02/2019 


#ifndef _BUS_CPP
#define _BUS_CPP

#include <iostream>
#include <vector>
#include "Unit.h"
#include "Connection.h"
#include "CollectionBusLine.h"
#include "Unit.h"
#include "Config.h"

class Bus : public Unit {

private:
    unsigned int n_bus_lines;  //Number of outputs from the bus
    unsigned int input_ports_bus_line;
    unsigned int connection_width;
    unsigned int fifo_size;
    std::vector<CollectionBusLine*> collection_bus_lines; 
    
public:
    Bus(id_t id, std::string name, Config stonne_cfg);
    unsigned int getNBusLines()    {return this->n_bus_lines;}
    unsigned int getInputPortsBusLine()   {return this->input_ports_bus_line;}
    std::vector<std::vector<Connection*>> getInputConnections();    
    std::vector<Connection*> getOutputConnections(); //Get the output connections of all the lines
    Connection* getInputConnectionFromBusLine(unsigned int busID, unsigned int inputID); //Get a specific inpur from a specific bus line


    void cycle(); //Get the inputs and send as many as posssible to the outputs
  
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    ~Bus();
   

    
};

#endif
