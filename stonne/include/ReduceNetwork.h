//
// Created by Francisco Munoz on 19/06/19.
//

#ifndef __REDUCENETWORK__H__
#define __REDUCENETWORK__H__

#include <iostream>
#include <fstream>
#include <map>
#include "types.h"
#include "Config.h"
#include "CompilerComponent.h"
#include "Connection.h"
#include "Unit.h"
#include "Tile.h"
#include "DNNLayer.h"

class ReduceNetwork : public Unit{
    
public:
    ReduceNetwork(id_t id, std::string name)  : Unit(id, name) {}
    virtual ~ReduceNetwork() {}
    virtual void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections) {assert(false);} //Connect all the memory ports from buses (busID, lineID) to its corresponding switches
    virtual std::map<int, Connection*> getLastLevelConnections() {assert(false);}
    virtual void setOutputConnection(Connection* outputConnection)  {assert(false);} //This function set the outputConnection with the Prefetch buffer
    virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding) {assert(false);}
    virtual void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size) {assert(false);}
    virtual void configureSignalsSortTree(adderconfig_t sort_configuration) {assert(false);}
    virtual void resetSignals() {assert(false);}


    //Cycle function
    virtual void cycle(){}
    
    virtual void printConfiguration(std::ofstream& out, unsigned int indent) {}  //This function prints the configuration of the ASNetwork (i.e., ASwitches configuration such as ADD_2_1, ADD_3_1, etc)
    virtual void printStats(std::ofstream& out, unsigned int indent) {} //This functions prints the statistics obtained during the execution. 
    virtual void printEnergy(std::ofstream& out, unsigned int indent){}
    

};

#endif 
