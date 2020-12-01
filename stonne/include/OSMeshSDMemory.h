#ifndef __OSMESHSDMEMORY__H__
#define __OSMESHSDMEMORY__H__

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
#include "MultiplierNetwork.h"
#include "ReduceNetwork.h"


class OSMeshSDMemory : public MemoryController {
private:
    DNNLayer* dnn_layer; // Layer loaded in the accelerator
    ReduceNetwork* reduce_network; //Reduce network used to be reconfigured
    MultiplierNetwork* multiplier_network; //Multiplier network used to be reconfigured

    unsigned int M;
    unsigned int N;

    unsigned int K;   //Number of columns MK matrix and rows KN matrix. Extracted from dnn_layer->get_C(); 

    unsigned int OUT_DIST_VN;  //To calculate the output memory address
    unsigned int OUT_DIST_VN_ITERATION; //To calculate the memory address
    Connection* write_connection;
    OSMeshControllerState current_state; //Stage to control what to do according to the state
    std::vector<SparseVN> configurationVNs; //A set of each VN size mapped onto the architecture.
    std::vector<unsigned int> vnat_table; //Every element is a VN, indicating the column that is calculating
    //Connection* read_connection;
    std::vector<Connection*> read_connections; //Input port connections. There are as many connections as n_read_ports are specified.
  
    //Input parameters
    unsigned int ms_rows;
    unsigned int ms_cols;
    unsigned int n_read_ports;
    unsigned int n_write_ports; 
    unsigned int write_buffer_capacity;
    unsigned int port_width;

    unsigned int rows_used;
    unsigned int cols_used;

    unsigned int ms_size_per_input_port;
    //Fifos
    Fifo* write_fifo; //Fifo uses to store the writes before going to the memory
    
    std::vector<Fifo*> input_fifos; //Fifos used to store the inputs before being fetched
    std::vector<Fifo*> psum_fifos; //Fifos used to store partial psums before being fetched
    //Fifo* read_fifo; //Fifo used to store the inputs before being fetched
    //Fifo* psums_fifo; //Fifo used to store partial psums before being fetched
 
    //Addresses
    address_t MK_address;
    address_t KN_address;
    address_t output_address;



    //Tile parameters
    unsigned int T_N;           //Actual value of T_N if adaptive tiling is used
    unsigned int T_K; //This is the actual value of tile of K. This is just 1 in this case
    unsigned int T_M;
    unsigned int iter_N;
    unsigned int iter_K;  
    unsigned int iter_M;
    
    //Current parameters
    unsigned int current_M;
    unsigned int current_N;
    unsigned int current_K;    
    

    //Signals
    bool configuration_done; //Indicates whether the architecture has been configured to perform the delivering
    bool execution_finished; //Flag that indicates when the execution is over. This happens when all the output values have been calculated.
    bool iteration_completed;
    
    bool metadata_loaded;   //Flag that indicates whether the metadata has been loaded 
    bool layer_loaded; //Flag that indicates whether the layer has been loaded.
   

   unsigned int current_output;
   unsigned int output_size; 

   unsigned int current_output_iteration;
   unsigned int n_iterations_completed;
   unsigned int output_size_iteration;

   //For stats
   std::vector<Connection*> write_port_connections; 
   cycles_t local_cycle;
   SDMemoryStats sdmemoryStats; //To track information
   
   //Aux functions
   void receive();
   void send();
   void sendPackageToInputFifos(DataPackage* pck);
   std::vector<Connection*> getWritePortConnections()    const {return this->write_port_connections;}
    
    
public:
    OSMeshSDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection);
    ~OSMeshSDMemory();
    void setLayer(DNNLayer* dnn_layer,  address_t KN_address, address_t MK_address, address_t output_address, Dataflow dataflow);
    void setTile(Tile* current_tile);
    void setReadConnections(std::vector<Connection*> read_connections);
    void setWriteConnections(std::vector<Connection*> write_port_connections); //All the write connections must be set at a time
    void cycle();
    bool isExecutionFinished();

    void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer) {assert(false);} // Supported by this controller
    void setDenseSpatialData(unsigned int T_N, unsigned int T_K) {assert(false);}
    void setReduceNetwork(ReduceNetwork* reduce_network) {this->reduce_network=reduce_network;}
    //Used to configure the MultiplierNetwork according to the controller
    void setMultiplierNetwork(MultiplierNetwork* multiplier_network) {this->multiplier_network = multiplier_network;}
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
};


#endif 
