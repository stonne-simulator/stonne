//
// Created by Francisco Munoz on 17/06/19.
//

#ifndef __SPARSEFLEXMSNETWORK__H__
#define __SPARSEFLEXMSNETWORK__H__

#include "DSNetwork.h"
#include "Connection.h"
#include "SparseFlex_MSwitch.h"
#include "DSwitch.h"
#include "Unit.h"
#include <iostream>
#include "CompilerMSN.h"
#include "Tile.h"
#include "DNNLayer.h"
#include "MultiplierNetwork.h"
#include "Stats.h"
#include "types.h"

#include <map>

class SparseFlex_MSNetwork : public MultiplierNetwork{
private:
    std::map<int, SparseFlex_MSwitch*> mswitchtable; //Table with the multiplier switches
    std::map<int, Connection*> fwconnectiontable; //Table with the forwarding connections. Each position is the input for the MS in that certain position. 
    // The connections between the DS and the RS is not needed. Once connected by the external MAERi object, the MSwitches and DSwitches communicate
    //each other
    unsigned int ms_size; // Number of multipliers
    unsigned int forwarding_ports; //SparseFlex_MSNetwork needs this parameter to create the network
    unsigned int buffers_capacity; //Capacity of the buffers in the MSwitches. This is neccesary to check if it is feasible to manage the folding.
    void virtualNetworkConfig(std::map<unsigned int,unsigned int> vn_conf); //set the VN of each MS
    void fwLinksConfig(std::map<unsigned int, bool> ms_fwsend_enabled, std::map<unsigned int, bool> ms_fwreceive_enabled); //Set to each MS if it must receive and/or send data from/to the fw link 
    void forwardingPsumConfig(std::map<unsigned int, bool> forwarding_psum_enabled); //Set to each MS if it has to act as a normal multiplier or an extra MS to accumulate psums

    void directForwardingPsumConfig(std::map<unsigned int, bool> direct_forwarding_psum_enabled); //Same as previous one, but the forwarding is always carried out without control

    void setPhysicalConnection(); //Create the forwarding links in this SparseFlex_MSNetwork
    void nWindowsConfig(unsigned int n_windows);
    void nFoldingConfig(std::map<unsigned int, unsigned int> n_folding_configuration); //Set number of folds for each MS
    void configurePartialGenerationMode(bool mergePartialSum);
    void configureForwarderMode();
    std::map<int, SparseFlex_MSwitch*> getMSwitches();
    std::map<int, Connection*> getForwardingConnections(); //Return the connections


public:
    /*
       By the default the implementation of the MS just receives a single element, calculate a single psum and/or send a single input activation to the neighbour. This way, the parameters
       input_ports, output_ports and forwarding_ports will be set as the single data size. If this implementation change for future tests, this can be change easily bu mofifying these three parameters.
     */
    SparseFlex_MSNetwork(id_t id, std::string name, Config stonne_cfg);
    ~SparseFlex_MSNetwork();
    //set connections from the distribution network to the multiplier network
    void setInputConnections(std::map<int, Connection*> input_connections);
    //Set connections from the Multiplier Network to the Reduction Network
    void setOutputConnections(std::map<int, Connection*> output_connections);
    //void setMemoryConnections(std::map<int, Connection*> memory_connections);
    void cycle(); 
    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding, multiplierconfig_t multiplierconfig);
    void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size);
    void resetSignals(); 
    void printConfiguration(std::ofstream& out, unsigned int indent);
    void printStats(std::ofstream &out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    void setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections);
    //MSNetworkStats getStats();

};
#endif 
