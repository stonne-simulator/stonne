//
// Created by Francisco Munoz on 17/06/19.
//

#ifndef __DSNETWORK__H__
#define __DSNETWORK__H__

#include "MSNetwork.h"
#include "DSwitch.h"
#include "Unit.h"
#include <iostream>
#include <fstream>
#include <map>
#include "Config.h"


#define CONNECTIONS_PER_SWITCH 2
#define LEFT 0
#define RIGHT 1


class DSNetwork : public Unit{
private:
    unsigned int ms_size; //Number of multipliers. i.e., the leaves of the network 
    unsigned int port_width;
    int nlevels; //Number of levels of the DS without taking into account the MS level
    std::map<std::pair<int, int>, DSwitch* > dswitchtable; //Map with the switches of the topology. The connection among them will be different depending on the topology used
    std::map<std::pair<int, int>, Connection*> connectiontable; // Outputs connections of each level. 
    Connection* inputConnection;  //Given by external
    
    
public:
    DSNetwork(id_t id, std::string name, Config stonne_cfg, unsigned int ms_size, Connection* inputConnection); //ms_size = ms_size of the group that contain this tree
    ~DSNetwork();
    const int getNLevels()  const { return this->nlevels; }
    const int getMsSize()   const { return this->ms_size; }
    std::map<int, Connection*> getLastLevelConnections();
    void setInputConnection(Connection* inputConnection)  { this->inputConnection = inputConnection; } //This function set the inputConnection with the Prefetch buffer

    //Useful functions

    //Cycle function
    void cycle();
    unsigned long get_time_routing();
    void printStats(std::ofstream& out, unsigned int indent); //Print the stats of the component
    void printEnergy(std::ofstream& out, unsigned int indent);
    //void setConfiguration(Config cfg);

    
    

};

#endif 
