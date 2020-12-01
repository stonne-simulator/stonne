//Created 13/06/2019

#ifndef __data_package_h__
#define __data_package_h__

#include "types.h"
#include <iostream>

/*

This class represents the wrapper of a certain data. It is used in both networks ART and DS but there are some fields that are used in just one particular class. For example, 
since the DS package does not need the VN, it is not used during that network. 

*/

class DataPackage {

private:
    //General field
    size_t size_package; //Actual size of the package. This just accounts for the truly data that is sent in a real implementation
    data_t data; //Data in the package
    operand_t data_type; //Type of data (i.e., WEIGHT, IACTIVATION, OACTIVATION, PSUM)
    id_t source; //Source that sent the package
   
    // Fields only used for the DS
    bool* dests;  // Used in multicast traffic to indicate the receivers
    unsigned int n_dests; //Number of receivers in multicast operation
    unsigned int unicast_dest; //Indicates the destination in case of unicast package
    traffic_t traffic_type; // IF UNICAST dest is unicast_dest. If multicast, dest is indicate using dests and n_dests.

    unsigned int VN; //Virtual network where the psum is found
    adderoperation_t operation_mode; //operation that got this psum (Comparation or SUM)
    
    unsigned int output_port; //Used in the psum package to get the output port that was used in the bus to send the data 
    unsigned int iteration_g; //Indicates the g value of this package (i.e., the number of g iteration). This is used to avoid sending packages of some iteration g and k without having performing the previous ones.
    unsigned int iteration_k; //Indicates the k value of this package (i.e, the number of k iteration). This is used to avoid sending packages of some iteration k (output channel k) without having performed the previous iterations yet
    

        
public:
    //General constructor to be reused in both types of packages
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source);
    
    //DS Package constructors for creating unicasts, multicasts and broadcasts packages
    //General constructor for DS
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type);
    // Unicast package constructor. 
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int unicast_dest);
    //Multicast package. dests must be dynamic memory since the array is not copied. 
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, bool* dests, unsigned int n_dests); //Constructor
    //Broadcast package
    //Needs nothing. Just indicates is the type broadcast

    //ART Package constructor (only one package for this type)
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source, unsigned int VN, adderoperation_t operation_mode);
    ~DataPackage();
    DataPackage(DataPackage* pck); //Constructor copy used to repeat a package
    //Getters
    const size_t get_size_package()            const {return this->size_package;}
    const data_t get_data()                    const {return this->data;}
    const operand_t get_data_type()            const {return this->data_type;}
    const id_t get_source()                    const {return this->source;}
    const traffic_t get_traffic_type()         const {return this->traffic_type;}
    bool isBroadcast()                   const {return this->traffic_type==BROADCAST;}
    bool isUnicast()                     const {return this->traffic_type==UNICAST;}
    bool isMulticast()                   const {return this->traffic_type==MULTICAST;}
    const bool* get_dests()                    const {return this->dests;}
    unsigned int get_unicast_dest()        const {return this->unicast_dest;}
    unsigned int get_n_dests()                  const {return this->n_dests;}
    unsigned int getOutputPort()           const {return this->output_port;}
    unsigned int getIterationK()           const {return this->iteration_k;}
    void setOutputPort(unsigned int output_port);
    void setIterationK(unsigned int iteration_k); //Used to control avoid a package from the next iteration without having calculated the previous ones.

    unsigned int get_vn()                  const {return this->VN;}
    adderoperation_t get_operation_mode()   const {return this->operation_mode;}
};

#endif
