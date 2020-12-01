// Created on 06/11/2019 by Francisco Munoz Martinez

#include "DSNetworkTop.h"
#include <iostream>
#include "utility.h"

DSNetworkTop::DSNetworkTop(id_t id, std::string name, Config stonne_cfg) : DistributionNetwork(id,name) {
    this->n_input_ports = stonne_cfg.m_SDMemoryCfg.n_read_ports;
    if(stonne_cfg.m_MSNetworkCfg.multiplier_network_type==LINEAR) {
        this->ms_size_per_port = stonne_cfg.m_MSNetworkCfg.ms_size / n_input_ports;
    }
    else if(stonne_cfg.m_MSNetworkCfg.multiplier_network_type==OS_MESH) {
        this->ms_size_per_port = (stonne_cfg.m_MSNetworkCfg.ms_rows + stonne_cfg.m_MSNetworkCfg.ms_cols) / n_input_ports;
    }
    this->port_width = stonne_cfg.m_DSwitchCfg.port_width;
    for(int i=0; i<this->n_input_ports; i++) {
        //Creating the top connection first
        Connection* conn = new Connection(this->port_width);
        //Creating the tree
        std::string name = "ASNetworkTree "+i; 
        DSNetwork* dsnet = new DSNetwork(i,name, stonne_cfg, this->ms_size_per_port, conn); //Creating the dsnetwork with the connection
        connections.push_back(conn);
        dsnetworks.push_back(dsnet);
       
    }

}

std::map<int, Connection*> DSNetworkTop::getLastLevelConnections() {
    std::map<int, Connection*> connectionsLastLevel; 
    for(int i=0; i<this->n_input_ports; i++)  { //For each tree we add its lastlevelconnections
        std::map<int, Connection*> connectionsPort = this->dsnetworks[i]->getLastLevelConnections(); //Getting the last level conns of the tree i
        unsigned int index_base = i*ms_size_per_port; //Current connection respect to the first connection in the first tree
        for(int j=0; j<this->ms_size_per_port; j++) { //We are sure connectionsPort size is ms_size_per_per_port 
            Connection* current_connection = connectionsPort[j]; //Local index
            connectionsLastLevel[index_base+j]=current_connection; //Adding to the global list
        }
    }
    return connectionsLastLevel;
}

//Return the top connections (i.e., the input connections that connects the DSMemory ports with the subtrees)
std::vector<Connection*> DSNetworkTop::getTopConnections() {
    return this->connections;
}

void DSNetworkTop::cycle() {
    for(int i=0; i<this->n_input_ports; i++) {
        dsnetworks[i]->cycle();
    }
}

DSNetworkTop::~DSNetworkTop() {
    for(int i=0; i<this->n_input_ports; i++) {
        delete dsnetworks[i];
        delete connections[i];
    }
}

void DSNetworkTop::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"DSNetworkStats\" : {" << std::endl;
    //General statistics if there are

    //For each subtree
    out << ind(indent+IND_SIZE) << "\"DSTreeStats\" : [" << std::endl;
    for(int i=0; i<this->n_input_ports; i++) {
        dsnetworks[i]->printStats(out, indent+IND_SIZE+IND_SIZE);
        if(i==(this->n_input_ports-1)) { //If I am in the last tree I do not have to separate the objects with a comma
            out << std::endl;
        }
        else { //Put a comma between two DStree objects
            out << "," << std::endl;
        }
    }
    out << ind(indent+IND_SIZE) << "]" << std::endl;
    out << ind(indent) << "}";
}

void DSNetworkTop::printEnergy(std::ofstream& out, unsigned int indent) {
   /*
       This component prints:
           - Connections between memory and DSNetworks
           - DSNetworks
   */
   
   //Printing wires between memory and DSNetwork
   for(int i=0; i<this->n_input_ports; i++)  {
       this->connections[i]->printEnergy(out, indent, "DN_WIRE");
   }

   //Printing ASNetworks
   for(int i=0; i<this->n_input_ports; i++)  {
       this->dsnetworks[i]->printEnergy(out, indent);
   }
   
}
