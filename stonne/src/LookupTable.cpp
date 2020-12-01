//Created by Francisco Munoz on 25/06/2019

#include "LookupTable.h"
#include <vector>
#include "DataPackage.h"

LookupTable::LookupTable(id_t id, std::string name, Config stonne_cfg, Connection* inputConnection, Connection* outputConnection) : Unit(id, name) {
    // Collecting parameters from the configuration file
    this->latency=stonne_cfg.m_LookUpTableCfg.latency;
    this->port_width = stonne_cfg.m_LookUpTableCfg.port_width;
    // End collecting parameters from the configuration file
    this->inputConnection = inputConnection;
    this->outputConnection = outputConnection;
}

void LookupTable::cycle() {
    if(this->inputConnection->existPendingData()) {
        //std::cout << "LookupTABLE is executing" << std::endl;
        std::vector<DataPackage*> pck_to_receive = this->inputConnection->receive(); 
        //TODO apply activation function
        this->outputConnection->send(pck_to_receive);
        //for(int i=0; i<pck_to_receive.size(); i++) {
        //    std::cout << "Data received: " << pck_to_receive[i]->get_data() << std::endl;
        //}
        
    }
}
