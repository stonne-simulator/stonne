#ifndef _COMPONENT_H_
#define _COMPONENT_H_

#include <string>
#include "Types.h"
#include "Connection.hpp"
#include <vector>

class Component {
private:
    string componentName;       // Name of the component
    bool enabled;               // This flag set if the device is on
    cycles_t idleCycles;        // Number of cycles in which the component is idle.
    cycles_t totalCycles;       // Amount of total cycles. This includes idleCycles.
    id_t id;
public:
    Component(id_t id, const string& componentName) {
        this->id = id;
        this->componentName = componentName;
        this->enabled = true;
        this->idleCycles = 0;
        this->totalCycles = 0;
    }

    const string& getComponentName() const       {return componentName;}
    const bool isEnabled() const                 {return enabled;}
    const cycles_t getIdleCycles() const         {return idleCycles;}
    const cycles_t getTotalCycles() const        {return totalCycles;}
    const id_t getId() const                     {return id;}

    virtual void cycle() = 0;
    virtual void printStats(ofstream& out, unsigned int indent) = 0;
    //virtual void reset() = 0;


    

    
};

#endif
