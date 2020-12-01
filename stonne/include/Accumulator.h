//Created 19/02/2020

#ifndef __ACCUMULATOR__h
#define __ACCUMULATOR__h

#include "types.h"
#include "DataPackage.h"
#include "Connection.h"
#include <vector>
#include "Fifo.h"
#include "Unit.h"
#include "Config.h"
#include "Stats.h"
/*
*/

class Accumulator : public Unit {
private:
    unsigned int input_ports;                        // Input port 
    unsigned int output_ports;                       // output port 
    unsigned int buffers_capacity;                   //Buffers size in bytes
    unsigned int port_width;                         //Bit width of each port

    unsigned int busID;                              //CollectionBus connected to this ASwitch
    unsigned int inputID;                            //Number of input of the Collection Bus busID connected to this AS.

    //Inputs fifos
    Fifo* input_fifo;                          // Array of packages that are received from the adders
   
    // Output Fifos
    Fifo* output_fifo;                      // Output fifo to the parent

    adderoperation_t operation_mode; //Adder or comp

    unsigned int current_capacity;                   // the capacity must not exceed the input bw of the connection
    Connection* inputConnection;                 // This is the input left connection of the Adder
    Connection* outputConnection;                    // This is the output connection of the adder

    cycles_t latency;                                  // Number of cycles to compute a sum. This is configurable since can vary depending on the implementation technology and frequency

    //Operation functions. This functions can be changed in order to perform different types of length operations
    DataPackage* perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right);    //Perform 2:1 sum

    cycles_t local_cycle;
    ASwitchStats aswitchStats; //To track the behaviour of the FEASwitch

    //Extensions
    DataPackage* temporal_register; //Temporal register to accumulate partial sums
    unsigned int n_psums; //Number of psums before accumulation
    unsigned int current_psum; //Current psum performed
    unsigned int n_accumulator;

    AccumulatorStats accumulatorStats; //Object to track the behaviour of the Accumulator


    //Private functions
    void route();


public:
    Accumulator(id_t id, std::string name, Config stonne_cfg, unsigned int n_accumulator); 
    Accumulator(id_t id, std::string name, Config stonne_cfg, unsigned int n_accumulator, Connection* inputConnection, Connection* outputConnection);
    ~Accumulator();

    //Connection setters
    void setInputConnection(Connection* inputLeftConnection);                       // Set the input left connection of the Adder
    void setOutputConnection(Connection* outputConnection);                             // Set the output connection of the Adder
    void setNPSums(unsigned int n_psums);
    void resetSignals();

    // Getters
    const unsigned int getNAcummulator()      const {return this->n_accumulator;}
    const unsigned int getOutputPorts() const {return this->output_ports;}                          // Get the output ports

    // Functionality
    void send(); //Packages of data to be sent depending on routing. 
    void receive(); //take data from connections


    void cycle();   //Computing a cycle. Based on routing the AS decides where the data goes. 

    void printConfiguration(std::ofstream& out, unsigned int indent);  //This function prints the configuration of FEASwitch such as  the operation mode, augmented link enabled, etc
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
	

};

#endif

