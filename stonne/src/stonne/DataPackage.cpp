//Created 13/06/2019

#include "DataPackage.hpp"
#include <assert.h>
#include <string.h>

//General constructor implementation

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, std::size_t row, std::size_t col) {
  this->m_size = size_package;
  this->m_data = data;
  this->m_dataType = data_type;
  this->m_source = source;
  this->m_trafficType = UNICAST;  //Default
  this->m_row = row;
  this->m_col = col;
}

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, traffic_t traffic_type, std::size_t row, std::size_t col)
    : DataPackage(size_package, data, data_type, source, row, col) {
  this->m_trafficType = traffic_type;
  this->p_dests = NULL;
}

// Unicast package constructor.
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, traffic_t traffic_type, std::size_t unicast_dest,
                         std::size_t row, std::size_t col)
    : DataPackage(size_package, data, data_type, source, traffic_type, row, col) {
  assert(traffic_type == UNICAST);
  this->m_unicastDest = unicast_dest;
}

//Multicast package. dests must be dynamic memory since the array is not copied.
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, traffic_t traffic_type, bool* dests, std::size_t n_dests,
                         std::size_t row, std::size_t col)
    : DataPackage(size_package, data, data_type, source, traffic_type, row, col) {
  this->p_dests = dests;
  this->m_nDests = n_dests;
}

//psum package
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, std::size_t VN, adderoperation_t operation_mode,
                         std::size_t row, std::size_t col)
    : DataPackage(size_package, data, data_type, source, row, col) {
  this->m_vn = VN;
  this->m_operationMode = operation_mode;
}

void DataPackage::setOutputPort(std::size_t output_port) {
  this->m_outputPort = output_port;
}

void DataPackage::setIterationK(std::size_t iteration_k) {
  this->m_iterationK = iteration_k;
}

//Copy constructor
DataPackage::DataPackage(DataPackage* pck) {
  this->m_size = pck->get_size_package();
  this->m_data = pck->get_data();
  this->m_dataType = pck->get_data_type();
  this->m_source = pck->get_source();
  this->m_trafficType = pck->get_traffic_type();
  this->m_unicastDest = pck->get_unicast_dest();
  this->m_vn = pck->get_vn();
  this->m_operationMode = pck->get_operation_mode();
  this->m_outputPort = m_outputPort;
  this->m_iterationK = pck->getIterationK();
  this->m_row = pck->getRow();
  this->m_col = pck->getCol();
  if (this->m_trafficType == MULTICAST) {
    this->m_nDests = pck->get_n_dests();
    const bool* prev_pck_dests = pck->get_dests();
    this->p_dests = new bool[this->m_nDests];
    //for(int i=0; i<n_dests; i++) {
    //    this->dests[i]=prev_pck_dests[i];
    //}
    memcpy(this->p_dests, prev_pck_dests, sizeof(bool) * this->m_nDests);
  }
}

DataPackage::~DataPackage() {

  if (this->m_trafficType == MULTICAST) {
    delete[] p_dests;
  }
}
