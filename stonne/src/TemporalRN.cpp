//
// Created by Francisco Munoz on 19/06/19.
//
#include "TemporalRN.h"
#include <assert.h>
#include "utility.h"
#include <math.h>

//TODO Conectar los enlaces intermedios de forwarding
//This Constructor creates the reduction tree similar to the one shown in the paper
TemporalRN::TemporalRN(id_t id, std::string name, Config stonne_cfg, Connection* outputConnection) : ReduceNetwork(id, name) {
    // Collecting the parameters from configuration file
    this->stonne_cfg = stonne_cfg;
    this->port_width=stonne_cfg.m_ASwitchCfg.port_width;
    //End collecting the parameters from the configuration file
    //
    //Calculating the number of accumulators based on the shape of the multiplier network
    if(stonne_cfg.m_MSNetworkCfg.multiplier_network_type==LINEAR) {
        this->accumulation_buffer_size = stonne_cfg.m_MSNetworkCfg.ms_size;
    }

    else {
        this->accumulation_buffer_size = stonne_cfg.m_MSNetworkCfg.ms_rows*stonne_cfg.m_MSNetworkCfg.ms_cols;
    }
    this->accumulationBuffer = new AccumulationBuffer(0, "AccumulationBuffer", this->stonne_cfg, this->accumulation_buffer_size);
    //Creating the input connections
    for(int i=0; i<this->accumulation_buffer_size; i++) {
        Connection* input_connection = new Connection(this->port_width);
        inputconnectiontable.push_back(input_connection);
    }
    this->accumulationBuffer->setInputConnections(inputconnectiontable);
    
        
}

TemporalRN::~TemporalRN() {
   delete this->accumulationBuffer;
   for(int i=0; i < inputconnectiontable.size(); i++) {
       delete this->inputconnectiontable[i];
   }




}


void TemporalRN::setMemoryConnections(std::vector<std::vector<Connection*>>  memoryConnections) {
    unsigned int n_bus_lines = memoryConnections.size();
    std::cout << "N_bus_lines: " << n_bus_lines << std::endl;

    for(int i=0; i<this->accumulation_buffer_size; i++) {
        unsigned  int inputID = (i / n_bus_lines);
        unsigned int busID = i % n_bus_lines;
        Connection* mem_conn = memoryConnections[busID][inputID];
        outputconnectiontable.push_back(mem_conn);
        std::cout << "ACCUMUlATOR " << i << " connected to BUS " << busID << " INPUT " << inputID << std::endl;
    }

    //Finally we connect the output links with the memory 
    this->accumulationBuffer->setMemoryConnections(outputconnectiontable);

}

std::map<int, Connection*> TemporalRN::getLastLevelConnections() {
    //Converting from vector to map for questions of compatibility with the rest of the code
    std::map<int, Connection*> last_level_connections; 
    for(int i=0; i<inputconnectiontable.size(); i++) {
        last_level_connections[i]=inputconnectiontable[i]; 
    }
    return last_level_connections;
}

void TemporalRN::resetSignals() {
    this->accumulationBuffer->resetSignals();
}

void TemporalRN::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding) {
    this->accumulationBuffer->configureSignals(current_tile, dnn_layer, ms_size, n_folding);

} 

void TemporalRN::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size) {
    assert(false);
}


//TODO Implementar esto bien
void TemporalRN::cycle() {
    this->accumulationBuffer->cycle();
}

//Print configuration of the TemporalRN 
void TemporalRN::printConfiguration(std::ofstream& out, unsigned int indent) {

    out << ind(indent) << "\"ASNetworkConfiguration\" : {" << std::endl;
    out << ind(indent) << "}";
    
}

//Printing stats
void TemporalRN::printStats(std::ofstream& out, unsigned int indent) {
     out << ind(indent) << "\"ASNetworkStats\" : {" << std::endl;
     this->accumulationBuffer->printStats(out, indent+IND_SIZE);
     out << ind(indent) << "}";
}

void TemporalRN::printEnergy(std::ofstream& out, unsigned int indent) {
     /*
      The TemporalRN component prints the counters for the next subcomponents:
          - Accumulators
          - Input wires are not printed as we consider this accumulator as inside the multiplier (it is an entire PE)

      Note that the wires that connect with memory are not taken into account in this component. This is done in the CollectionBus.

     */
     //Printing the accumulator stats
     this->accumulationBuffer->printEnergy(out, indent);

     

}
