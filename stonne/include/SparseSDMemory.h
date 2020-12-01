#ifndef __SPARSESDMEMORY__H__
#define __SPARSESDMEMORY__H__

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


class SparseSDMemory : public MemoryController {
private:
    DNNLayer* dnn_layer; // Layer loaded in the accelerator
    ReduceNetwork* reduce_network; //Reduce network used to be reconfigured
    MultiplierNetwork* multiplier_network; //Multiplier network used to be reconfigured

    unsigned int M;
    unsigned int N;
    Dataflow dataflow;
    unsigned int dim_sta;   //Number of vectors sta matrix. Extracted from dnn_layer->get_K(); (See equivalence with CNN)
    unsigned int K;   //Number of columns MK matrix and rows KN matrix. Extracted from dnn_layer->get_S(); 
    unsigned int dim_str;   //Number of vectors str matrix. Extracted from dnn_layer->get_N()
    unsigned int STA_DIST_ELEM;  //Distance in bitmap memory between two elements of the same vector
    unsigned int STA_DIST_VECTOR; //Disctance in bitmap memory between two elements of differ vectors.

    unsigned int STR_DIST_ELEM;   //Idem than before but with the STR matrix
    unsigned int STR_DIST_VECTOR;

    unsigned int OUT_DIST_VN;  //To calculate the output memory address
    unsigned int OUT_DIST_VN_ITERATION; //To calculate the memory address
    Connection* write_connection;
    SparsityControllerState current_state; //Stage to control what to do according to the state
    std::vector<SparseVN> configurationVNs; //A set of each VN size mapped onto the architecture.
    std::vector<unsigned int> vnat_table; //Every element is a VN, indicating the column that is calculating
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
    address_t STA_address;
    address_t STR_address;
    address_t output_address;



    //Metadata addresses
    metadata_address_t STA_metadata;
    metadata_address_t STR_metadata;
    metadata_address_t output_metadata;

    //Counters to calculate SRC and DST
    unsigned int* sta_counters_table; //Matrix of size rows*columns to figure out the dst of each sta value
    unsigned int* str_counters_table; //Matrix of size rows*columns of the str matrix to calculate the source of each bit enabled.

    //Pointers
    unsigned int str_current_index; //Streaming current index to calculate the next values to stream 
    unsigned int sta_current_index_metadata; //Stationary matrix current index (e.g., row in MK)
    unsigned int sta_current_index_matrix; //Index to next element in the sparse matrix
    unsigned int sta_current_j_metadata; //Index to current element in the same cluster. Used to manage folding
    unsigned int sta_last_j_metadata;  //Indext to last element in the same cluster. Used to manage folding
    //the boundaries of a certain fold is sta_current_j_metadata and sta_last_j_metadata

    //Signals
    bool configuration_done; //Indicates whether the architecture has been configured to perform the delivering
    bool stationary_distributed; //Indicates if the stationary values has been distributed for a certain iteration
    bool stationary_finished; //Flag that indicates that all the stationary values have been delivered
    bool stream_finished;  //Flag that indicates that all the streaming values have been delivered
    bool execution_finished; //Flag that indicates when the execution is over. This happens when all the output values have been calculated.
    bool sta_iter_completed; //Indicates if the pending psums have been writen back

    
    bool metadata_loaded;   //Flag that indicates whether the metadata has been loaded 
    bool layer_loaded; //Flag that indicates whether the layer has been loaded.
   

   unsigned int current_output;
   unsigned int output_size; 

   unsigned int current_output_iteration;
   unsigned int output_size_iteration;

   //For stats
   unsigned int n_ones_sta_matrix;
   unsigned int n_ones_str_matrix;
   std::vector<Connection*> write_port_connections; 
   cycles_t local_cycle;
   SDMemoryStats sdmemoryStats; //To track information
   
   //Aux functions
   void receive();
   void send();
   void sendPackageToInputFifos(DataPackage* pck);
   std::vector<Connection*> getWritePortConnections()    const {return this->write_port_connections;}
    
    
public:
    SparseSDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection);
    ~SparseSDMemory();
    void setLayer(DNNLayer* dnn_layer,  address_t KN_address, address_t MK_address, address_t output_address, Dataflow dataflow);
    void setTile(Tile* current_tile) {assert(false);}
    void setReadConnections(std::vector<Connection*> read_connections);
    void setWriteConnections(std::vector<Connection*> write_port_connections); //All the write connections must be set at a time
    void cycle();
    bool isExecutionFinished();

    void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata); // Supported by this controller
    void setReduceNetwork(ReduceNetwork* reduce_network) {this->reduce_network=reduce_network;}
    //Used to configure the MultiplierNetwork according to the controller
    void setMultiplierNetwork(MultiplierNetwork* multiplier_network) {this->multiplier_network = multiplier_network;}
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
};


#endif //SPARSESDMEMORY_H_
