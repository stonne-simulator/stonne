//Created 13/06/2019

#include "DataPackage.h"
#include <assert.h>
#include <string.h>

//General constructor implementation

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source, unsigned int row, unsigned int col) {
    this->size_package = size_package;
    this->data = data;
    this->data_type =data_type;
    this->source = source;
    this->traffic_type = UNICAST; //Default
    this->row = row;
    this->col = col;
}

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int row, unsigned int col) : DataPackage(size_package, data, data_type, source, row, col) {
    this->traffic_type = traffic_type;
    this->dests = NULL;
}
// Unicast package constructor. 
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int unicast_dest, unsigned int row, unsigned int col) : 
DataPackage(size_package, data, data_type, source, traffic_type, row, col) {
    assert(traffic_type == UNICAST);
    this->unicast_dest = unicast_dest;
}
//Multicast package. dests must be dynamic memory since the array is not copied. 
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, bool* dests, unsigned int n_dests, unsigned int row, unsigned int col) : DataPackage(size_package, data, data_type, source, traffic_type, row, col) {
    this->dests = dests;
    this->n_dests = n_dests;
}

//psum package
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source, unsigned int VN, adderoperation_t operation_mode, unsigned int row, unsigned int col): DataPackage(size_package, data, data_type, source, row, col) {
    this->VN = VN;
    this->operation_mode = operation_mode;
}

void DataPackage::setOutputPort(unsigned int output_port) {
    this->output_port = output_port;
}

void DataPackage::setIterationK(unsigned int iteration_k) {
    this->iteration_k = iteration_k;
}

//Copy constructor
DataPackage::DataPackage(DataPackage* pck) {
    this->size_package = pck->get_size_package();
    this->data = pck->get_data();
    this->data_type = pck->get_data_type();
    this->source = pck->get_source();
    this->traffic_type = pck->get_traffic_type();
    this->unicast_dest = pck->get_unicast_dest();
    this->VN = pck->get_vn();
    this->operation_mode = pck->get_operation_mode();
    this->output_port = output_port;
    this->iteration_k=pck->getIterationK();
    this->row=pck->getRow();
    this->col=pck->getCol();
    if(this->traffic_type == MULTICAST) {
        this->n_dests = pck->get_n_dests();  
        const bool* prev_pck_dests = pck->get_dests();
        this->dests = new bool[this->n_dests];
        //for(int i=0; i<n_dests; i++) {
        //    this->dests[i]=prev_pck_dests[i];
        //}
        memcpy(this->dests, prev_pck_dests, sizeof(bool)*this->n_dests);

    }

}

DataPackage::~DataPackage() {
  
    if(this->traffic_type==MULTICAST) {
        delete[] dests;
    }
}


