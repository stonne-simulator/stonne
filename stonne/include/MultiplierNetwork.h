
#ifndef __MULTIPLIERNETWORK__H__
#define __MULTIPLIERNETWORK__H__

#include "Connection.h"
#include "MSwitch.h"
#include "DSwitch.h"
#include "Unit.h"
#include <iostream>
#include "CompilerMSN.h"
#include "Tile.h"
#include "DNNLayer.h"
#include <assert.h>

#include <map>

class MultiplierNetwork : public Unit{
public:
    /*
       By the default the implementation of the MS just receives a single element, calculate a single psum and/or send a single input activation to the neighbour. This way, the parameters
       input_ports, output_ports and forwarding_ports will be set as the single data size. If this implementation change for future tests, this can be change easily bu mofifying these three parameters.
     */
    MultiplierNetwork(id_t id, std::string name) : Unit(id, name){}
    virtual ~MultiplierNetwork() {}
    //set connections from the distribution network to the multiplier network
    virtual void setInputConnections(std::map<int, Connection*> input_connections) {assert(false);}
    virtual void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections) {assert(false);}
    //Set connections from the Multiplier Network to the Reduction Network
    virtual void setOutputConnections(std::map<int, Connection*> output_connections) {assert(false);}
    virtual void cycle() {assert(false);}
    virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding) {assert(false);}
    virtual void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size) {assert(false);}
    virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding, multiplierconfig_t multiplierconfig) {assert(false);}

    virtual void resetSignals() {assert(false);}
    virtual void printConfiguration(std::ofstream& out, unsigned int indent) {assert(false);}
    virtual void printStats(std::ofstream &out, unsigned int indent) {assert(false);}
    virtual void printEnergy(std::ofstream& out, unsigned int indent) {assert(false);}
};
#endif 
