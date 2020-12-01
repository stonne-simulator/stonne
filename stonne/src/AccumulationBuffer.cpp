//
// Created by Francisco Munoz on 19/06/19.
//
#include "AccumulationBuffer.h"
#include <assert.h>
#include "utility.h"
#include <math.h>

//TODO Conectar los enlaces intermedios de forwarding
//This Constructor creates the reduction tree similar to the one shown in the paper
AccumulationBuffer::AccumulationBuffer(id_t id, std::string name, Config stonne_cfg, unsigned int n_accumulators) : Unit(id, name) {
    // Collecting the parameters from configuration file
    this->port_width=stonne_cfg.m_ASwitchCfg.port_width;
    this->n_accumulators = n_accumulators; //Number of accumulators
    std::string name_str = "accumulator ";
    for(int i=0; i<this->n_accumulators; i++) {
	std::string name_acc = name_str+=i;
        Accumulator* acc = new Accumulator(i, name_acc, stonne_cfg, i);
        accumulatortable[i]=acc;
    }
     

}

AccumulationBuffer::~AccumulationBuffer() {
    for(int i=0; i<this->n_accumulators; i++) {
        delete accumulatortable[i];
    }
}


void AccumulationBuffer::setMemoryConnections(std::vector<Connection*>  memoryConnections) {
    for(int i=0; i<this->n_accumulators; i++) {
        outputconnectiontable[i]=memoryConnections[i];
	accumulatortable[i]->setOutputConnection(memoryConnections[i]);
    } 
}

void AccumulationBuffer::setInputConnections(std::vector<Connection*> inputConnections) {
    for(int i=0; i<this->n_accumulators; i++) {
        inputconnectiontable[i]=inputConnections[i];
	accumulatortable[i]->setInputConnection(inputConnections[i]);
    }
}

void AccumulationBuffer::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding) {

    this->NPSumsConfiguration(n_folding); //All the accumulation buffers have the same folding iteration numbers which means that in this case all the VNs are similar

} 

void AccumulationBuffer::resetSignals() {
     for(std::map<int, Accumulator*>::iterator it=accumulatortable.begin(); it != accumulatortable.end(); ++it) {
        it->second->resetSignals();
    }

}

void AccumulationBuffer::NPSumsConfiguration(unsigned int n_psums) {
     for(std::map<int, Accumulator*>::iterator it=accumulatortable.begin(); it != accumulatortable.end(); ++it) {
        it->second->setNPSums(n_psums);
    }


}

void AccumulationBuffer::cycle() {
  for(int i=0; i<this->n_accumulators; i++) {
      accumulatortable[i]->cycle();
  } 
}

//Print configuration of the ASNetwork 
void AccumulationBuffer::printConfiguration(std::ofstream& out, unsigned int indent) {

}

//Printing stats
void AccumulationBuffer::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"AccumulationBufferStats\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"AccumulatorStats\" : [" << std::endl;   
        for(int i=0; i < this->n_accumulators; i++) {  
                Accumulator* ac = accumulatortable[i];
                ac->printStats(out, indent+IND_SIZE+IND_SIZE);
                if(i==(this->n_accumulators-1)) {  //If I am in the last accumulator, the comma to separate the accumulators is not added
                    out << std::endl; //This is added because the call to acc print do not show it (to be able to put the comma, if neccesary)
                }
                else {
                    out << "," << std::endl; //Comma and line break are added to separate with the next accumulator in the array
                }



        }
        out << ind(indent+IND_SIZE) << "]" << std::endl;
    out << ind(indent) << "}";

}

void AccumulationBuffer::printEnergy(std::ofstream& out, unsigned int indent) {
     /*
      The AccumulationBuffer component prints the counters for the next subcomponents:
          - Accumualators
          - wires that connect each accumulator with its AdderSwitch

      Note that the wires that connect with memory are not taken into account in this component. This is done in the CollectionBus.

     */

     //Printing the input wires
     
     for(std::map<int, Connection*>::iterator it=inputconnectiontable.begin(); it != inputconnectiontable.end(); ++it) {
         Connection* conn = inputconnectiontable[it->first];
         conn->printEnergy(out, indent, "RN_WIRE");
     }

    
    //Printing the Accumulator energy stats and their fifos stats
     for(std::map<int,Accumulator*>::iterator it=accumulatortable.begin(); it != accumulatortable.end(); ++it) {
        Accumulator* acc = accumulatortable[it->first]; //index
        acc->printEnergy(out, indent); //Setting the direction
     }


}
