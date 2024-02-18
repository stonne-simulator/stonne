#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <iostream>
#include "common/types.hpp"

//--------------------------------------------------------------------
// DSNetework Configuration Parameters
//--------------------------------------------------------------------
class DSNetworkConfig {
 public:
  std::size_t n_switches_traversed_by_cycle{23};  //TODO Not implemented yet. Value from paper

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// DSwitch Configuration Parameters
//--------------------------------------------------------------------
class DSwitchConfig {
 public:
  //By the moment there is nothing to configure for the DSwitch
  std::size_t latency{1};       //Actually is less than 1. We do not implement this either
  std::size_t input_ports{1};   //Number of input_ports. By default this will be 1
  std::size_t output_ports{2};  //Number of output ports. By default this will be 2
  std::size_t port_width{16};   //Bit width

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// MSNetwork Configuration Parameters
//--------------------------------------------------------------------
class MSNetworkConfig {
 public:
  MultiplierNetwork_t multiplier_network_type{LINEAR};
  std::size_t ms_size{64};  //Number of multiplier switches.
  std::size_t ms_rows{0};   // not initialized
  std::size_t ms_cols{0};   // not initialized

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// MSwitch Configuration Parameters
//--------------------------------------------------------------------
class MSwitchConfig {
 public:
  cycles_t latency{1};          //Latency of the MS to perform a multiplication. This number is expressed in number of cycles. //TODO not implemented
  std::size_t input_ports{1};   //Number of input ports of the MS. This number is 1 by default in MAERI
  std::size_t output_ports{1};  // Number of output ports of the MS.
  std::size_t forwarding_ports{
      1};  // Number of forwarding ports of the MS. This is basically the number of elements that can be forwarded in a single cycle and in MAERI architecture is just 1.
  std::size_t port_width{16};  //Bit width
  std::size_t buffers_capacity{
      2048};  //Number of elements that can be stored in the MS buffers. TODO In future implementations this could be splited up, taking each buffer capacity in a different parameter.

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// ASNetwork Configuration Parameters
//--------------------------------------------------------------------
class ASNetworkConfig {
 public:
  ReduceNetwork_t reduce_network_type{ASNETWORK};  //Type of the ReduceNetwork configured in this moment
  std::size_t accumulation_buffer_enabled{0};
  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// ASwitch Configuration Parameters
//--------------------------------------------------------------------
class ASwitchConfig {
 public:
  std::size_t buffers_capacity{256};
  std::size_t input_ports{2};       //Number of input ports of the ASwitch. By  default in MAERI this is just 2
  std::size_t output_ports{1};      //Number of output ports of the ASwitch. By default  in MAERI this is 1
  std::size_t forwarding_ports{1};  //Nuber of forwarding ports of the ASwitch.
  std::size_t port_width{16};       //Bit width
  cycles_t latency{1};              //Latency of the AS to perform. This number is expressed in number of cycles. //TODO To implement

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// LookUpTable Configuration Parameters
//--------------------------------------------------------------------
class LookUpTableConfig {
 public:
  cycles_t latency{1};  //Latency of the LookUpTable to perform. This number must be expressed in number of cycles. 0 no supported. //TODO To implement
  std::size_t port_width{1};

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// SDMemory Controller Configuration Parameters
//--------------------------------------------------------------------
class SDMemoryConfig {
 public:
  MemoryController_t mem_controller_type{MAERI_DENSE_WORKLOAD};  //Memory controller type (e.g., DENSE_WORKLOAD or SPARSE_GEMM)
  std::size_t write_buffer_capacity{256};                        //Capacity of the buffers expressed in terms of number of elements
  std::size_t n_read_ports{4};                                   //dn_bw
  std::size_t n_write_ports{4};                                  //rn_bw
  std::size_t port_width{16};                                    //Bit width
  uint64_t weight_address{0};                                    //Address where the weights are first stored
  uint64_t input_address{10000};                                 //Address ehere the inputs are first stored
  uint64_t output_address{20000};                                //Address where the output are first stored
  uint32_t data_width{4};                                        //Number of bytes allocated to each data element.
  uint32_t n_write_mshr{16};                                     // Number of parallel write requests that can be done in parallel

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
//--------------------------------------------------------------------
//MAIN CONFIGURATION OBJECT
//--------------------------------------------------------------------
//--------------------------------------------------------------------

class Config {
 public:
  //General parameters
  std::size_t print_stats_enabled{1};  //Specified whether the statistics must be printed.

  //DSNetwork Configuration
  DSNetworkConfig m_DSNetworkCfg;

  //DSwitch Configuration
  DSwitchConfig m_DSwitchCfg;

  //MSNetwork Configuration
  MSNetworkConfig m_MSNetworkCfg;

  //MSwitch Configuration
  MSwitchConfig m_MSwitchCfg;

  //ASNetwork Configuration
  ASNetworkConfig m_ASNetworkCfg;

  //ASwitch Configuration
  ASwitchConfig m_ASwitchCfg;

  //LookUpTableConfiguration
  LookUpTableConfig m_LookUpTableCfg;

  //SDMemory controller configuration
  SDMemoryConfig m_SDMemoryCfg;

  //Load parameters from configuration file using TOML Syntax
  void loadFile(std::string config_file);

  //print the configuration parameters
  void printConfiguration(std::ofstream& out, std::size_t indent);

  //Indicates whether according to the hardware parameters, sparsity is enabled in the architecture
  bool sparsitySupportEnabled();

  //Indicates whether according to the hardware parameters, the operation of CONV itself is supported. Otherwise, the operation can be done
  //using GEMM operation
  bool convOperationSupported();
};

#endif
