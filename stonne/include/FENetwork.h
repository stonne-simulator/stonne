//
// Created by Francisco Munoz on 19/02/2020.
//

#ifndef __FENETWORK__H__
#define __FENETWORK__H__

#include "MSNetwork.h"
#include "FEASwitch.h"
#include <iostream>
#include <fstream>
#include <map>
#include "types.h"
#include "Config.h"
#include "CompilerFEN.h"
#include "ReduceNetwork.h"

#define CONNECTIONS_PER_SWITCH 2
#define LEFT 0
#define RIGHT 1


class FENetwork : public ReduceNetwork {
private:
    unsigned int port_width; //Width in bits of each port
    unsigned int ms_size; //Number of multipliers. i.e., the leaves of the network 
    int nlevels; //Number of levels of the AS without taking into account the MS level
    std::map<std::pair<int, int>, FEASwitch* > aswitchtable; //Map with the switches of the topology. The connection among them will be different depending on the topology used

    //Copy of the pointers of the map aswitchtable used to generate the connections between the ART and the bus in the same way as the implementation in bluespec does.
    std::vector<FEASwitch*> single_switches; //List of switches that are single reduction switches  (see blueSpec implementation. i.e., the do not have forwarding connections)
    std::vector<FEASwitch*> double_switches; //List of double switches that are double reduction switches (i.e., in bluespec implementation these are sw that have fw links).

    std::map<std::pair<int, int>, Connection*> inputconnectiontable; // input connections of each level. 
    std::map<std::pair<int, int>, Connection*> forwardingconnectiontable; // Forwarding connections of each level (intermedium links)
    std::map<std::pair<int, int>, Connection*> foldingconnectiontable; //Forwarding connections between each node and its folding manager
    Connection* outputConnection;  //Given by external
    
    
public:
    FENetwork(id_t id, std::string name, Config stonne_cfg, Connection* output_connection);
    ~FENetwork();
    const int getNLevels()  const { return this->nlevels; }
    const int getMsSize()   const { return this->ms_size; }
    std::vector<FEASwitch*> getSingleSwitches() {return this->single_switches;}
    std::vector<FEASwitch*> getDoubleSwitches() {return this->double_switches;}
    void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections); //Connect all the memory ports (busID, lineID) to its corresponding AS
    std::map<int, Connection*> getLastLevelConnections();
    void setOutputConnection(Connection* outputConnection)  { this->outputConnection = outputConnection; } //This function set the outputConnection with the Prefetch buffer
    void addersConfiguration(std::map<std::pair<int, int>, adderconfig_t> adder_configurations);
    void forwardingConfiguration(std::map<std::pair<int,int>, fl_t> fl_configurations); //Configure the forwarding links. Enable the required ones. 
    void childsLinksConfiguration(std::map<std::pair<int,int>, std::pair<bool,bool>> childs_configuration);
    void forwardingToMemoryConfiguration(std::map<std::pair<int,int>, bool> forwarding_to_memory_enabled);
    void forwardingToFoldNodeConfiguration(std::map<std::pair<int,int>, bool> forwarding_to_fold_node_enabled);
    void NPSumsConfiguration(unsigned int n_psums);

    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding);


    //Cycle function
    void cycle();
    
    void printConfiguration(std::ofstream& out, unsigned int indent);  //This function prints the configuration of the ASNetwork (i.e., FEASwitches configuration such as ADD_2_1, ADD_3_1, etc)
    void printStats(std::ofstream& out, unsigned int indent); //This functions prints the statistics obtained during the execution. 
    void printEnergy(std::ofstream& out, unsigned int indent);
    

};

#endif 
