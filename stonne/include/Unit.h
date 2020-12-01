#ifndef _UNIT_h_
#define _UNIT_h_

#include <iostream>
#include <string>
#include <fstream>
#include "Config.h"

class Unit {
private:
    id_t id;  //Id of the component
    std::string name; //Name of the component

public:
    Unit(id_t id, std::string name) {
        this->id=id;
        this->name=name;
    }

    virtual void printStats(std::ofstream& out, unsigned int indent) {} //Print the stats of the component
    virtual void printEnergy(std::ofstream& out, unsigned int indent) {} //Print the counters to get the consumption of the unit
    virtual void cycle() {} //Execute a cycle in the component
    virtual void setConfiguration(Config cfg) {} //set the configuration parameters of the component
};


#endif
