#ifndef __OUTERSPGEMMSDMEMORY__H__
#define __OUTERSPGEMMSDMEMORY__H__

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
#include "Memory.h"

class OuterLoopSpGEMMSDMemory : public MemoryController {
private:
    DNNLayer* dnn_layer; // Layer loaded in the accelerator
    ReduceNetwork* reduce_network; //Reduce network used to be reconfigured
    MultiplierNetwork* multiplier_network; //Multiplier network used to be reconfigured

    std::vector<std::queue<DataPackage*>>* intermediate_memory;
    std::vector<std::queue<DataPackage*>> swap_memory; //To be used during the several iterations
    std::vector<std::queue<DataPackage*>>* pointer_current_memory; //This is to exchange between a particular row in intermediate_memory and swap_memory
    std::vector<std::queue<DataPackage*>>* pointer_next_memory;

    unsigned int M;
    unsigned int N;
    unsigned int K;   //Number of columns MK matrix and rows KN matrix. Extracted from dnn_layer->get_C(); 

    Connection* write_connection;
    SparsityControllerState current_state; //Stage to control what to do according to the state
    std::vector<SparseVN> configurationVNs; //A set of each VN size mapped onto the architecture.
    std::vector<int> vnat_table;
    std::vector<int> ms_group;
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
    address_t MK_address;
    address_t KN_address;
    address_t output_address;



    //Metadata addresses
    metadata_address_t MK_row_id;
    metadata_address_t MK_col_pointer; //Actually this is col pointer, but the functionality is the same.
    metadata_address_t KN_col_id;
    metadata_address_t KN_row_pointer; 

    //SST Memory hierarchy component structures and variables
    Memory<float>& mem;

    //Current pointers
    unsigned int current_MK;
    unsigned int current_MK_col_pointer;
    unsigned int current_MK_row_id;
    unsigned int current_KN;
    unsigned int current_KN_row_pointer;
    unsigned int current_KN_col_id;

    /* SST variables */
    uint64_t weight_dram_location;
    uint64_t input_dram_location;
    uint64_t output_dram_location;

    uint32_t data_width;
    uint32_t n_write_mshr;
   
    //Aux parameters
    unsigned int MK_number_nnz; 
    unsigned int multipliers_used;
    unsigned int n_str_data_sent;
    unsigned int n_str_data_received;

    //Signals
    bool configuration_done; //Indicates whether the architecture has been configured to perform the delivering
    bool stationary_distributed; //Indicates if the stationary values has been distributed for a certain iteration
    bool stationary_finished; //Flag that indicates that all the stationary values have been delivered
    bool stream_finished;  //Flag that indicates that all the streaming values have been delivered
    bool execution_finished; //Flag that indicates when the execution is over. This happens when all the output values have been calculated.
    bool sta_iter_completed; //Indicates if the pending psums have been writen back
    bool last_sta_iteration_completed;
    bool STA_complete;
    bool STR_complete;
    bool multiplication_phase_finished;
    bool sort_down_last_iteration_finished;
    bool sort_down_iteration_finished; 
    bool sort_up_iteration_finished;
    bool sort_up_received_first_value;
    bool sort_up_exception_row_empty;
    
    
    bool metadata_loaded;   //Flag that indicates whether the metadata has been loaded 
    bool layer_loaded; //Flag that indicates whether the layer has been loaded.
   

   unsigned int current_output;
   unsigned int output_size; 

   unsigned int current_output_iteration;
   unsigned int output_size_iteration;

   //SORTING TREE CONTROL
   unsigned int sort_col_id;
   unsigned int sort_row_id;
   unsigned int sort_sub_block_id;
   unsigned int sort_num_blocks;
   bool swap_memory_enabled; 
   //For stats
   unsigned int n_ones_sta_matrix;
   unsigned int n_ones_str_matrix;
   std::vector<Connection*> write_port_connections; 
   cycles_t local_cycle;
   SDMemoryStats sdmemoryStats; //To track information

   Tile* tile; //Not really used in sparseflex

   //Variable to manage the number of sorting iterations
   int sorting_iterations;
   int current_sorting_iteration;
   int n_values_stored;
   
   //Aux functions
   void receive();
   void send();
   bool doLoad(uint64_t addr, DataPackage* data_package);
   bool doStore(uint64_t addr, DataPackage* data_package);
   void sendPackageToInputFifos(DataPackage* pck);
   std::vector<Connection*> getWritePortConnections()    const {return this->write_port_connections;}


    
    
public:
    OuterLoopSpGEMMSDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection, Memory<float>& mem);
    ~OuterLoopSpGEMMSDMemory();
    void setLayer(DNNLayer* dnn_layer,  address_t KN_address, address_t MK_address, address_t output_address, Dataflow dataflow);
    void setTile(Tile* current_tile) {assert(false);}
    void setReadConnections(std::vector<Connection*> read_connections);
    void setWriteConnections(std::vector<Connection*> write_port_connections); //All the write connections must be set at a time
    void cycle();
    bool isExecutionFinished();

    void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, metadata_address_t KN_metadata_id, metadata_address_t KN_metadata_pointer);
    void setReduceNetwork(ReduceNetwork* reduce_network) {this->reduce_network=reduce_network;}
    //Used to configure the MultiplierNetwork according to the controller
    void setMultiplierNetwork(MultiplierNetwork* multiplier_network) {this->multiplier_network = multiplier_network;}
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    SDMemoryStats getStats() {return this->sdmemoryStats;}

};


#endif //SPARSESDMEMORY_H_
