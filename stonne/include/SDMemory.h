//Created by Francisco Munoz Martinez on 02/07/2019
#ifndef __SDMEMORY__H__
#define __SDMEMORY__H__

#include <list>
#include "Tile.h"
#include "Connection.h"
#include "Fifo.h"
#include "types.h"
#include "DNNLayer.h"
#include "Unit.h"
#include "Config.h"
#include "DataPackage.h"
#include "Stats.h"
#include "MemoryController.h"

//This class contains for each VN the next address to write 
class VNAT_Register {
public:
    unsigned int VN; //VN Saved
    unsigned int base_addr; //Base addr of this VN (i.e., the first element to compute).
    unsigned int addr; //Offset
    unsigned int current_N;
    unsigned int current_G;
    unsigned int current_K;
    unsigned int current_X;
    unsigned int current_Y; 
    unsigned int current_R;
    unsigned int current_S;
    unsigned int current_C;
    //To calculate next output_address
    unsigned int iter_N; 
    unsigned int iter_G;
    unsigned int iter_K;
    unsigned int iter_X;
    unsigned int iter_Y;
    unsigned int iter_R;
    unsigned int iter_S;
    unsigned int iter_C;
    unsigned int n_psums; //psums per window
    unsigned int current_psum;
    DNNLayer* dnn_layer;
    Tile* current_tile;
    bool finished;
  
   
    VNAT_Register(unsigned int VN, unsigned int addr, unsigned int N, unsigned int G, unsigned int K, unsigned int X, unsigned int Y,  
    unsigned int iter_N, unsigned int iter_G, unsigned int iter_K, unsigned int iter_X, unsigned int iter_Y, unsigned int iter_R, unsigned int iter_S, unsigned int iter_C, DNNLayer* dnn_layer, Tile* current_tile);
    void update(); //Update variables to the next cycle 
    unsigned int get_address();
    
};

class SDMemory : public MemoryController {
private:
    DNNLayer* dnn_layer; // Layer loaded in the accelerator
    Tile* current_tile;  // Layer loaded in the tile
    ReduceNetwork* reduce_network; //This is not used in this controller as the configuration is performed in STONNEModel when the tile is loaded, and this is needed just once
    MultiplierNetwork* multiplier_network; //Idem as reduce_network
    Connection* write_connection;
    //Connection* read_connection;
    std::vector<Connection*> read_connections; //Input port connections. There are as many connections as n_read_ports are specified.
  
    //Input parameters
    unsigned int num_ms;
    unsigned int n_read_ports;
    unsigned int n_write_ports; 
    unsigned int write_buffer_capacity;
    unsigned int port_width;

    unsigned int ms_size_per_input_port;
    //Fifos
    Fifo* write_fifo; //Fifo uses to store the writes before going to the memory
    
    std::vector<Fifo*> input_fifos; //Fifos used to store the inputs before being fetched
    std::vector<Fifo*> psum_fifos; //Fifos used to store partial psums before being fetched
    //Fifo* read_fifo; //Fifo used to store the inputs before being fetched
    //Fifo* psums_fifo; //Fifo used to store partial psums before being fetched
 
    //Addresses
    address_t filter_address;
    address_t input_address;
    address_t output_address;

    //Signals
    bool weights_distributed; //Indicates if the weights have been distributed for a certain iteration
    bool fw_link_enabled; //Indicates if the fw link is enabled in this cycle and therefore the number of bw used per cycle is less
    bool weights_finished; //Flag that indicates that all the weights have been delivered
    bool input_finished;  //Flag that indicates that all the inputs have been delivered
    bool tile_loaded; //SPecify if the tile is loaded
    bool execution_finished; //Flag that indicates when the execution is over. This happens when all the opixels have been calculated.
   
    //Variables to track the progress of the execution
    unsigned int iter_R;
    unsigned int iter_S;
    unsigned int iter_C;
    unsigned int iter_G;
    unsigned int iter_N;
    unsigned int iter_K;
    unsigned int iter_X;
    unsigned int iter_Y;

    unsigned int current_R;
    unsigned int current_S;
    unsigned int current_C;   
    unsigned int current_G;
    unsigned int current_N;
    unsigned int current_K;
    unsigned int current_X;
    unsigned int current_Y;

    //Variable to track the number of opixels calculated 
    unsigned int current_output_pixel;  //This variable has the count for the current number of output pixels calculated
    unsigned int output_pixels_to_compute;  //This variable has the number of output pixels that the simulator must calculate before finishing the execution
    unsigned int output_psums_per_channel;

    //Variables to make the calculation easier
    unsigned int channel_filter_size;
    unsigned int row_filter_size;
    unsigned int filter_size;
    unsigned int channel_input_size;
    unsigned int row_input_size;
    unsigned int input_size;
    unsigned int channel_output_size;
    unsigned int row_output_size;
    unsigned int output_size;
    unsigned int group_size;

    std::list<DataPackage*> packages_created; // Vector used to track the packages and delete them at the end of the execution
   std::vector<Connection*> write_port_connections; 
   VNAT_Register** VNAT;  //VNAT with as many registers as VN configured in the accelerator
   cycles_t local_cycle;
   SDMemoryStats sdmemoryStats; //To track information
   
   //Aux functions
   void receive();
   void sendPackageToInputFifos(DataPackage* pck);
   void send();
   std::vector<Connection*> getWritePortConnections()    const {return this->write_port_connections;}
    
    
public:
    SDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection);
    ~SDMemory();
    void setLayer(DNNLayer* dnn_layer,  address_t input_address, address_t filter_address, address_t output_address, Dataflow dataflow);
    void setTile(Tile* current_tile);
    void setReadConnections(std::vector<Connection*> read_connections);
    void setWriteConnections(std::vector<Connection*> write_port_connections); //All the write connections must be set at a time
    void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) {assert(false);} //Not supported by this controller
    void setReduceNetwork(ReduceNetwork* reduce_network) {this->reduce_network=reduce_network;}
    //Used to configure the MultiplierNetwork according to the controller if needed
    void setMultiplierNetwork(MultiplierNetwork* multiplier_network) {this->multiplier_network = multiplier_network;}


    void cycle();
    bool isExecutionFinished();
   
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
};


#endif //SDMEMORY_H_
