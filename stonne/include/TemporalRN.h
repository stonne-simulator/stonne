
#ifndef __TEMPORALREDUCTIONNETWORK__H__
#define __TEMPORALREDUCTIONNETWORK__H__

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



class TemporalRN : public ReduceNetwork {
private:
    unsigned int port_width; //Width in bits of each port
    std::vector<Connection*> inputconnectiontable; //Connections to the accumulation buffer 
    std::vector<Connection*> outputconnectiontable; //Connection to the collection bus
    Connection* outputConnection;  //Given by external
    AccumulationBuffer* accumulationBuffer; //Array of accumulators to perform the folding accumulation
    unsigned int accumulation_buffer_size; //Number of accumulation elements in the RN
    Config stonne_cfg;
    
    
public:
    TemporalRN(id_t id, std::string name, Config stonne_cfg, Connection* output_connection);
    ~TemporalRN();
    void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections); //Connect all the memory ports (busID, lineID) to its corresponding accumulator
    std::map<int, Connection*> getLastLevelConnections();
    void setOutputConnection(Connection* outputConnection)  { this->outputConnection = outputConnection; } //This function set the outputConnection with the Prefetch buffer
    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding);
    void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size);
    void resetSignals();


    //Cycle function
    void cycle();
    
    void printConfiguration(std::ofstream& out, unsigned int indent);  
    void printStats(std::ofstream& out, unsigned int indent); 
    void printEnergy(std::ofstream& out, unsigned int indent);
    

};

#endif 
