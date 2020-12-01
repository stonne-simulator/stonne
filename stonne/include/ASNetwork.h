//
// Created by Francisco Munoz on 19/06/19.
//

#ifndef __ASNETWORK__H__
#define __ASNETWORK__H__

#include "MSNetwork.h"
#include "ASwitch.h"
#include <iostream>
#include <fstream>
#include <map>
#include "types.h"
#include "Config.h"
#include "CompilerART.h"
#include "ReduceNetwork.h"
#include "AccumulationBuffer.h"

#define CONNECTIONS_PER_SWITCH 2
#define LEFT 0
#define RIGHT 1


class ASNetwork : public ReduceNetwork {
private:
    unsigned int port_width; //Width in bits of each port
    unsigned int ms_size; //Number of multipliers. i.e., the leaves of the network 
    int nlevels; //Number of levels of the AS without taking into account the MS level
    std::map<std::pair<int, int>, ASwitch* > aswitchtable; //Map with the switches of the topology. The connection among them will be different depending on the topology used

    //Copy of the pointers of the map aswitchtable used to generate the connections between the ART and the bus in the same way as the implementation in bluespec does.
    std::vector<ASwitch*> single_switches; //List of switches that are single reduction switches  (see blueSpec implementation. i.e., the do not have forwarding connections)
    std::vector<ASwitch*> double_switches; //List of double switches that are double reduction switches (i.e., in bluespec implementation these are sw that have fw links).

    std::map<std::pair<int, int>, Connection*> inputconnectiontable; // input connections of each level. 
    std::map<std::pair<int, int>, Connection*> forwardingconnectiontable; // Forwarding connections of each level (intermedium links)
    std::vector<Connection*> accumulationbufferconnectiontable; //Connections to the accumulation buffer if it is used
    Connection* outputConnection;  //Given by external
    AccumulationBuffer* accumulationBuffer; //Array of accumulators to perform the folding accumulation
    bool accumulation_buffer_enabled;
    Config stonne_cfg;
    
    
public:
    ASNetwork(id_t id, std::string name, Config stonne_cfg, Connection* output_connection);
    ~ASNetwork();
    const int getNLevels()  const { return this->nlevels; }
    const int getMsSize()   const { return this->ms_size; }
    std::vector<ASwitch*> getSingleSwitches() {return this->single_switches;}
    std::vector<ASwitch*> getDoubleSwitches() {return this->double_switches;}
    void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections); //Connect all the memory ports (busID, lineID) to its corresponding AS
    std::map<int, Connection*> getLastLevelConnections();
    void setOutputConnection(Connection* outputConnection)  { this->outputConnection = outputConnection; } //This function set the outputConnection with the Prefetch buffer
    void addersConfiguration(std::map<std::pair<int, int>, adderconfig_t> adder_configurations);
    void forwardingConfiguration(std::map<std::pair<int,int>, fl_t> fl_configurations); //Configure the forwarding links. Enable the required ones. 
    void childsLinksConfiguration(std::map<std::pair<int,int>, std::pair<bool,bool>> childs_configuration);
    void forwardingToMemoryConfiguration(std::map<std::pair<int,int>, bool> forwarding_to_memory_enabled);
    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding);
    void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size);
    void resetSignals();


    //Cycle function
    void cycle();
    
    void printConfiguration(std::ofstream& out, unsigned int indent);  //This function prints the configuration of the ASNetwork (i.e., ASwitches configuration such as ADD_2_1, ADD_3_1, etc)
    void printStats(std::ofstream& out, unsigned int indent); //This functions prints the statistics obtained during the execution. 
    void printEnergy(std::ofstream& out, unsigned int indent);
    

};

#endif 
