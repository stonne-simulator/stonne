//
// Created by Francisco Munoz on 17/06/19.
//

#ifndef __OSMeshMN__H__
#define __OSMeshMN__H__


#include "DSNetwork.h"
#include "Connection.h"
#include "MultiplierOS.h"
#include "DSwitch.h"
#include "Unit.h"
#include <iostream>
#include "CompilerMultiplierMesh.h"
#include "Tile.h"
#include "DNNLayer.h"
#include "MultiplierNetwork.h"

#include <map>

class OSMeshMN : public MultiplierNetwork{
private:
    std::map<std::pair<int, int>, MultiplierOS* > mswitchtable;
    std::map<std::pair<int, int>,  Connection*> verticalconnectiontable; //Table with the vertical connections
    std::map<std::pair<int, int>,  Connection*> horizontalconnectiontable; //Table with the horizontal connections
    std::map<std::pair<int, int>,  Connection*> accbufferconnectiontable; //Table with the accbuff connections
    unsigned int ms_rows; // Number of rows in the ms array
    unsigned int ms_cols; // Number of columns in the ms array
    unsigned int forwarding_ports; //MSNetwork needs this parameter to create the network
    unsigned int buffers_capacity; //Capacity of the buffers in the MSwitches. This is neccesary to check if it is feasible to manage the folding.
    unsigned int port_width; //Not used yet

    void setPhysicalConnection(); //Create the links
    std::map<std::pair<int,int>, MultiplierOS*> getMSwitches();
    //std::map<std::pair<int,int>, Connection*> getTopConnections(); //Return the connections
    


public:
    /*
       By the default the implementation of the MS just receives a single element, calculate a single psum and/or send a single input activation to the neighbour. This way, the parameters
       input_ports, output_ports and forwarding_ports will be set as the single data size. If this implementation change for future tests, this can be change easily bu mofifying these three parameters.
     */
    OSMeshMN(id_t id, std::string name, Config stonne_cfg);
    ~OSMeshMN();
    //set connections from the distribution network to the multiplier network
    void setInputConnections(std::map<int, Connection*> input_connections);
    //Set connections from the Multiplier Network to the Reduction Network
    void setOutputConnections(std::map<int, Connection*> output_connections);
    void cycle(); 
    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding);
    void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size);
    void resetSignals(); 
    void printConfiguration(std::ofstream& out, unsigned int indent);
    void printStats(std::ofstream &out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
};
#endif 
