//Created by Francisco Munoz Martinez on 02/07/2019
#ifndef __MEMORYCONTROLLER__H__
#define __MEMORYCONTROLLER__H__

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
#include <assert.h>
#include "ReduceNetwork.h"
#include "MultiplierNetwork.h"

class MemoryController : Unit {        
public:
    MemoryController(id_t id, std::string name) : Unit(id, name){}
    virtual void setLayer(DNNLayer* dnn_layer,  address_t input_address, address_t filter_address, address_t output_address, Dataflow dataflow) {assert(false);}
    virtual void setTile(Tile* current_tile) {assert(false);}
    virtual void setReadConnections(std::vector<Connection*> read_connections) {assert(false);}
    virtual void setWriteConnections(std::vector<Connection*> write_port_connections) {assert(false);} //All the write connections must be set at a time
    virtual void setSparseMetadata(metadata_address_t MK_metadata, metadata_address_t KN_metadata, metadata_address_t output_metadata) {assert(false);}
    virtual void setValueBasedMetadata(unsigned int* filter_metadata) {assert(false);};
    //Used to configure the ReduceNetwork according to the controller if needed
    virtual void setReduceNetwork(ReduceNetwork* reduce_network) {assert(false);}
    //Used to configure the MultiplierNetwork according to the controller if needed
    virtual void setMultiplierNetwork(MultiplierNetwork* multiplier_network) {assert(false);} 
    virtual void cycle() {assert(false);}
    virtual bool isExecutionFinished() {assert(false);}
    virtual void setDenseSpatialData(unsigned int T_N, unsigned int T_K){assert(false);}
    virtual void setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer){assert(false);}

   
    virtual void printStats(std::ofstream& out, unsigned int indent) {assert(false);}
    virtual void printEnergy(std::ofstream& out, unsigned int indent) {assert(false);}
};


#endif //SDMEMORY_H_
