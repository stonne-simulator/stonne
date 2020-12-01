//Created by Francisco Munoz Martinez on 25/06/2019

#ifndef __lookuptable__h
#define __lookuptable__h

#include "Connection.h"
#include "Unit.h"
#include "Config.h"
#include <iostream>

class LookupTable : Unit {
private:
    Connection* inputConnection; //From the ART
    Connection* outputConnection; //Torwards the memory
    cycles_t latency;
    unsigned int port_width;
public:
    LookupTable(id_t id, std::string name, Config stonne_cfg, Connection* inputConnection, Connection* outputConnection);
    void cycle();
};


#endif
