//Created by Francisco Munoz on 28/02/2019
#include <iostream>
#include "CollectionBus.h"
#include "utility.h"

Bus::Bus(id_t id, std::string name, Config stonne_cfg) : Unit(id, name) {
   this->n_bus_lines=stonne_cfg.m_SDMemoryCfg.n_write_ports;
   this->input_ports_bus_line=(stonne_cfg.m_MSNetworkCfg.ms_size / this->n_bus_lines)+ 1;
   this->connection_width = stonne_cfg.m_SDMemoryCfg.port_width;
   this->fifo_size = 100; //TODO 
   for(int i=0; i<n_bus_lines; i++) {
       std::string name="CollectionBusLine "+i;
       CollectionBusLine* busline = new CollectionBusLine(i, name, i, this->input_ports_bus_line, this->connection_width, this->fifo_size); 
       collection_bus_lines.push_back(busline);
   }
}

Bus::~Bus() {
    for(int i=0; i<n_bus_lines; i++) {
       delete collection_bus_lines[i]; 
    }
}

std::vector<std::vector<Connection*>> Bus::getInputConnections() {
    std::vector<std::vector<Connection*>> connections;
    for(int i=0; i<n_bus_lines; i++) {  //For each bus line we get all their connections and put them into the structure 
        CollectionBusLine* collection_bus = collection_bus_lines[i];
        std::vector<Connection*> connections_collection_bus = collection_bus->getInputConnections();
        connections.push_back(connections_collection_bus); //Setting the connections for that busID
    }
    return connections;
}

Connection* Bus::getInputConnectionFromBusLine(unsigned int busID, unsigned int inputID) {
    return this->collection_bus_lines[busID]->getInputPort(inputID);
}

std::vector<Connection*> Bus::getOutputConnections() {
    std::vector<Connection*> output_connections;
    for(int i=0; i<collection_bus_lines.size(); i++) {
        output_connections.push_back(collection_bus_lines[i]->getOutputPort());
    }
    return output_connections;
}

void Bus::cycle() { 
    for(int i=0; i<collection_bus_lines.size(); i++) {
        collection_bus_lines[i]->cycle();
    }
}

void Bus::printStats(std::ofstream &out, unsigned int indent) {
    out << ind(indent) << "\"CollectionBusStats\" : {" << std::endl;
        //out << ind(indent+IND_SIZE) << "\"ms_size\" : " << this->ms_size  << std::endl; DSNetwork global parameters
        out << ind(indent+IND_SIZE) << "\"CollectionBusLineStats\" : [" << std::endl;   //One entry per BusLine
        for(int i=0; i < this->collection_bus_lines.size(); i++) {  
                collection_bus_lines[i]->printStats(out, indent+IND_SIZE+IND_SIZE);
                if(i==(this->collection_bus_lines.size()-1)) {  //If I am in the last BusLine, the comma to separate the BusLines is not added
                    out << std::endl; //This is added because the call to ms print do not show it (to be able to put the comma, if neccesary)
                }
                else {
                    out << "," << std::endl; 
                }



        }
        out << ind(indent+IND_SIZE) << "]" << std::endl;
    out << ind(indent) << "}";

}

void Bus::printEnergy(std::ofstream& out, unsigned int indent) {
  /*
       This component prints the counters for each bus line 
  */

  for(int i=0; i < this->collection_bus_lines.size(); i++) {
      collection_bus_lines[i]->printEnergy(out, indent);
  }
}
