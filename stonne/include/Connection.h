//Created 13/06/2019

#ifndef __Connection__h
#define __Connection__h

#include "types.h"
#include "DataPackage.h"
#include <vector>
#include "Stats.h"

/*
This class Connection does not need ACK responses since in the accelerator the values are sent without a need of a request. Everything is controlled
by the control of the accelerator. 
*/


class Connection {
private:
    bool pending_data;   // Indicates if data exists
    size_t bw;           // Size in bytes of actual data. In the simulator this size is greater since we wrap the data into wrappers to track.
    std::vector<DataPackage*> data;   // Array of packages that are send/receive in  a certain cycle. The number of packages depends on the bw of the connection
    unsigned int current_capacity; // the capacity must not exceed the bw of the connection
    ConnectionStats connectionStats; //Tracking parameters

public:
    Connection(int bw);
    void send(std::vector<DataPackage*> data); //Package of data to be send. The sum of all the size_package of each package must not be greater than bw.
    std::vector<DataPackage*> receive();  //Receive a  packages from the connection
    bool existPendingData();
    
    void printEnergy(std::ofstream &out, unsigned int indent, std::string wire_type);


};


#endif

