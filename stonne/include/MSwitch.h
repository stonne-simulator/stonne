//Created 13/06/2019

#ifndef __MSwitch__h
#define __MSwitch__h

#include "types.h"
#include "DataPackage.h"
#include "Connection.h"
#include "Fifo.h"
#include "Unit.h"
#include <vector>
#include "Config.h"
#include "Stats.h"
/*
*/

class MSwitch : public Unit {
private:
    bool pending_to_compute;   // Indicates there is data pending to compute
    bool pending_output;
    Fifo* activation_fifo;   // Package received by the DN
    Fifo* forwarding_input_fifo; //Package received from the neighbour
    Fifo* forwarding_output_fifo; //Packages accumulated to be sent to the fw links when required
    Fifo* weight_fifo; //Weights stored in the MS
    Fifo* psum_fifo; //Psum ready to be sent to the parent

    std::vector<DataPackage*> psums_created; // All the psums created by this multiplier used to delete the package after the execution is finished.
    Connection* outputConnection;  // Towards the Reduce Network
    Connection* inputConnection; //From the DistributionNetwork
    Connection* outputForwardingConnection; //To the neighbour MS
    Connection* inputForwardingConnection; //Input from the neighbour
    cycles_t  latency;  //latency in number of cycles
    int num;

    //This values are in esence the size of a single element in the architecture (by default)
    unsigned int input_ports;
    unsigned int output_ports;
    unsigned int forwarding_ports;
    unsigned int buffers_capacity;
    unsigned int port_width; 
   
    //Signals
    int VN;
    bool inputForwardingEnabled;    //Control signal that specifies if the input fw link is enabled to receive data
    bool outputForwardingEnabled;   //Control signal that specifies if the output fw link is enabled to send data
    unsigned int n_windows;   //Control number that specifies the number of slides (shifts) a MS takes to calculate one row. It is useful to know if the MS has to send/receive a data from the fw link in a specific cycle (the first cycle of a row cannot receive data from neighbours since there is no data of that row.
    unsigned int n_folding; //Control the number of partial sums that must be generated to accumulate a whole ofmap value. if n_folding is 1, then partial sums is not required.

    //Counters to perform the control
    unsigned int current_n_windows; //Measure the number of windows that have been performed in the current row. 
    unsigned int current_n_folding; // meausre the number of foldings that have been performed in the current window. This help to know whether to read from the input or from the fw link
    bool forward_psum;  //Indicates if the behaviour of this MS is to forward a psum. This is useful to implement folding 
    bool direct_forward_psum; //Always forward the psum. It is different than the variable forward_psum as the last one has some control regarding the number of iterations. 
   
    cycles_t local_cycle;
    MSwitchStats mswitchStats; //Object to track the behaviour of the MSwitch

public:
    MSwitch(id_t id, std::string name, int num, Config stonne_cfg);
    MSwitch(id_t id, std::string name, int num, Config stonne_cfg, Connection* outputConnection, 
                                                                                                                                      Connection* inputConnection);
    ~MSwitch();
    void setOutputConnection(Connection* outputConnection); //Set the output connection of the switch (TO THE ADDER)
    void setInputForwardingConnection(Connection* inputForwardingConnection); //Set the right connection of the switch
    void setOutputForwardingConnection(Connection* outputForwardingConnection);
    void setInputConnection(Connection* inputConnection); //Set the input connection of the switch
    void send(); //Send the result through the outputConnection
    void receive(Connection* connection);  //Receive a package from the inputConnection or the forwarding connection and store it in this->data
    void forward(DataPackage* activation);
    void setVirtualNeuron(unsigned int VN); //Indicates the VN ID assigned 
    void setInputForwardingEnabled(bool inputForwardingEnabled); //Indicates if the MSwitch receives data from the fw link (from the RIGHT MS)
    void setOutputForwardingEnabled(bool outputForwardingEnabled); //Indicates if the MSwitch send data to the fw link (to the LEFT MS)
    void setNWindows(unsigned int n_windows); //The number of windows per row which is T_Y_ 
    void setNFolding(unsigned int n_folding); //The number of partial sums used to accumulate a whole sum
    void setForwardPsum(bool forward_psum); // Disable multipliplier function and enable the psum forwarding with the control managed by the multiplier
    void setDirectForwardPsum(bool direct_forward_psum); //Disable multiplier function and enable the psum forwarding. In this case, the psum is always forwarded (SIGMA).

    DataPackage* perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right); //Perform multiplication and returns result.
    
    void cycle(); //Computing a cyclels
    void resetSignals(); 

    void printConfiguration(std::ofstream& out, unsigned int indent);  //This function prints the configuration of MSwitch such us the VN ID
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);

	

};

#endif

