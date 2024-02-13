//Created 13/06/2019

#ifndef __data_package_h__
#define __data_package_h__

#include <iostream>
#include "types.hpp"

/*

This class represents the wrapper of a certain data. It is used in both networks ART and DS but there are some fields that are used in just one particular class. For example, 
since the DS package does not need the VN, it is not used during that network. 

*/

class DataPackage {

 private:
  //General field
  std::size_t m_size{};  //Actual size of the package. This just accounts for the truly data that is sent in a real implementation
  data_t m_data{};       //Data in the package
  std::size_t m_col{};
  std::size_t m_row{};
  operand_t m_dataType{};  //Type of data (i.e., WEIGHT, IACTIVATION, OACTIVATION, PSUM)
  stonne_id_t m_source{};  //Source that sent the package

  // Fields only used for the DS
  bool* p_dests{};              // Used in multicast traffic to indicate the receivers
  std::size_t m_nDests{};       //Number of receivers in multicast operation
  std::size_t m_unicastDest{};  //Indicates the destination in case of unicast package
  uint64_t m_address{};         //Indicate the address for the outputs
  traffic_t m_trafficType{};    // IF UNICAST dest is unicast_dest. If multicast, dest is indicate using dests and n_dests.

  std::size_t m_vn{};                  //Virtual network where the psum is found
  adderoperation_t m_operationMode{};  //operation that got this psum (Comparation or SUM)

  std::size_t m_outputPort{};  //Used in the psum package to get the output port that was used in the bus to send the data
  std::size_t
      m_iterationG{};  //Indicates the g value of this package (i.e., the number of g iteration). This is used to avoid sending packages of some iteration g and k without having performing the previous ones.
  std::size_t
      m_iterationK{};  //Indicates the k value of this package (i.e, the number of k iteration). This is used to avoid sending packages of some iteration k (output channel k) without having performed the previous iterations yet

 public:
  //General constructor to be reused in both types of packages
  DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, std::size_t row, std::size_t col);

  //DS Package constructors for creating unicasts, multicasts and broadcasts packages
  //General constructor for DS
  DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, traffic_t traffic_type, std::size_t row, std::size_t col);
  // Unicast package constructor.
  DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, traffic_t traffic_type, std::size_t unicast_dest, std::size_t row,
              std::size_t col);

  //Multicast package. dests must be dynamic memory since the array is not copied.
  DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, traffic_t traffic_type, bool* dests, std::size_t n_dests,
              std::size_t row,
              std::size_t col);  //Constructor
  //Broadcast package
  //Needs nothing. Just indicates is the type broadcast

  //ART Package constructor (only one package for this type)
  DataPackage(size_t size_package, data_t data, operand_t data_type, stonne_id_t source, std::size_t VN, adderoperation_t operation_mode, std::size_t row,
              std::size_t col);
  ~DataPackage();
  explicit DataPackage(DataPackage* pck);  //Constructor copy used to repeat a package

  //Getters
  [[nodiscard]] std::size_t get_size_package() const { return this->m_size; }

  [[nodiscard]] data_t get_data() const { return this->m_data; }

  void set_data(data_t data) { this->m_data = data; }

  void set_row(std::size_t row) { this->m_row = row; }

  void set_col(std::size_t col) { this->m_col = col; }

  [[nodiscard]] operand_t get_data_type() const { return this->m_dataType; }

  [[nodiscard]] stonne_id_t get_source() const { return this->m_source; }

  [[nodiscard]] traffic_t get_traffic_type() const { return this->m_trafficType; }

  void set_address(uint64_t addr) { this->m_address = addr; }

  [[nodiscard]] uint64_t get_address() const { return this->m_address; }

  bool isBroadcast() const { return this->m_trafficType == BROADCAST; }

  bool isUnicast() const { return this->m_trafficType == UNICAST; }

  bool isMulticast() const { return this->m_trafficType == MULTICAST; }

  const bool* get_dests() const { return this->p_dests; }

  std::size_t get_unicast_dest() const { return this->m_unicastDest; }

  std::size_t get_n_dests() const { return this->m_nDests; }

  std::size_t getOutputPort() const { return this->m_outputPort; }

  std::size_t getIterationK() const { return this->m_iterationK; }

  std::size_t getRow() const { return this->m_row; }

  std::size_t getCol() const { return this->m_col; }

  void setOutputPort(std::size_t output_port);
  void setIterationK(std::size_t iteration_k);  //Used to control avoid a package from the next iteration without having calculated the previous ones.

  std::size_t get_vn() const { return this->m_vn; }

  adderoperation_t get_operation_mode() const { return this->m_operationMode; }
};

#endif
