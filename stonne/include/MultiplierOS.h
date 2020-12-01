//Created 27/10/2020

#ifndef __MultiplierOS__h
#define __MultiplierOS__h

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

class MultiplierOS : public Unit {
private:
    Fifo* top_fifo;   // Packages received from top (i.e., weights)
    Fifo* left_fifo; //Packages recieved from legt (i.e., activations)
    Fifo* right_fifo; //Packages to be sent to the right (i.e., activations)
    Fifo* bottom_fifo; //Packages to be sent to the bottom (i.e., weights)
    Fifo* accbuffer_fifo; //Psum ready to be sent to the parent

    Connection* left_connection;  // To the left neighbour or memory port
    Connection* right_connection; //To the right neighbour
    Connection* top_connection; //To the top neighbour or the memory port
    Connection* bottom_connection; //Input from the neighbour 
    Connection* accbuffer_connection; //To the accbuffer to keep OS
    cycles_t  latency;  //latency in number of cycles
    int row_num;
    int col_num;
    int num; //General num, just used for information (num = row_num*ms_cols + col_num)
    //This values are in esence the size of a single element in the architecture (by default)
    unsigned int input_ports;
    unsigned int output_ports;
    unsigned int forwarding_ports;
    unsigned int buffers_capacity;
    unsigned int port_width; 
    unsigned int ms_rows;
    unsigned int ms_cols;
   
   
    cycles_t local_cycle;
    MultiplierOSStats mswitchStats; //Object to track the behaviour of the MSwitch

    //Signals
    unsigned int VN;
    bool forward_right=false; //Based on rows (windows) left and dimensions
    bool forward_bottom=false;

public:
    MultiplierOS(id_t id, std::string name, int row_num, int col_num, Config stonne_cfg);
    MultiplierOS(id_t id, std::string name, int row_num, int col_num, Config stonne_cfg, Connection* left_connection, Connection* right_connection, Connection* top_connection, Connection* bottom_connection); 
    ~MultiplierOS();
    void setTopConnection(Connection* top_connection); //Set the top connection 
    void setLeftConnection(Connection* left_connection); //Set the left connection
    void setRightConnection(Connection* right_connection);  //Set the right connection
    void setBottomConnection(Connection* bottom_connection);
    void setAccBufferConnection(Connection* accbuffer_connection);

    void send(); //Send right, bottom and psum fifos
    void receive();  //Receive from top and left

    DataPackage* perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right); //Perform multiplication and returns result.
    
    void cycle(); //Computing a cyclels
    void resetSignals(); 

    //Configure the forwarding signals that indicate if this ms has to forward data to the bottom and or right neighbours
    void configureBottomSignal(bool bottom_signal); 
    void configureRightSignal(bool right_signal); 
    void setVirtualNeuron(unsigned int VN);

    void printConfiguration(std::ofstream& out, unsigned int indent);  //This function prints the configuration of MSwitch such us the VN ID
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);

	

};

#endif

