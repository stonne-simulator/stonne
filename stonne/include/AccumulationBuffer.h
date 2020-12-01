//
// Created by Francisco Munoz on 29/06/2020.
//

#ifndef __ACCUMULATIONBUFFER__H__
#define __ACCUMULATIONBUFFER__H__

#include "Accumulator.h"
#include <iostream>
#include <fstream>
#include <map>
#include "types.h"
#include "Config.h"
#include "Tile.h"
#include "DNNLayer.h"



class AccumulationBuffer : public Unit {
private:
    unsigned int port_width; //Width in bits of each port
    unsigned int n_accumulators; //Number of accumulator array
    std::map<int, Accumulator* > accumulatortable; //Map with the accumulators

    std::map<int, Connection*> inputconnectiontable; // input connections
    std::map<int, Connection*> outputconnectiontable; // Output connections
public:
    AccumulationBuffer(id_t id, std::string name, Config stonne_cfg, unsigned int n_accumulators);
    ~AccumulationBuffer();
    void setMemoryConnections(std::vector<Connection*>  memoryConnections);
    void setInputConnections(std::vector<Connection*> inputConnections);
    void resetSignals();
    void NPSumsConfiguration(unsigned int n_psums);

    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding);
    //Cycle function
    void cycle();
    
    void printConfiguration(std::ofstream& out, unsigned int indent); 
    void printStats(std::ofstream& out, unsigned int indent); //This functions prints the stats
    void printEnergy(std::ofstream& out, unsigned int indent); //Print the counters
    

};

#endif 
