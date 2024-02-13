#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <iostream>
#include "types.hpp"

//--------------------------------------------------------------------
// DSNetework Configuration Parameters
//--------------------------------------------------------------------
class DSNetworkConfig {
 public:
  std::size_t n_switches_traversed_by_cycle;  //TODO Not implemented yet

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// DSwitch Configuration Parameters
//--------------------------------------------------------------------
class DSwitchConfig {
 public:
  //By the moment there is nothing to configure for the DSwitch
  std::size_t latency;
  std::size_t input_ports;   //Number of input_ports. By default this will be 1
  std::size_t output_ports;  //Number of output ports. By default this will be 2
  std::size_t port_width;    //Bit width

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// MSNetwork Configuration Parameters
//--------------------------------------------------------------------
class MSNetworkConfig {
 public:
  MultiplierNetwork_t multiplier_network_type;
  std::size_t ms_size;  //Number of multiplier switches.
  std::size_t ms_rows;
  std::size_t ms_cols;

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// MSwitch Configuration Parameters
//--------------------------------------------------------------------
class MSwitchConfig {
 public:
  cycles_t latency;          //Latency of the MS to perform a multiplication. This number is expressed in number of cycles. //TODO To imple
  std::size_t input_ports;   //Number of input ports of the MS. This number is 1 by default in MAERI
  std::size_t output_ports;  // Number of output ports of the MS.
  std::size_t
      forwarding_ports;  // Number of forwarding ports of the MS. This is basically the number of elements that can be forwarded in a single cycle and in MAERI architecture is just 1.
  std::size_t port_width;  //Bit width
  std::size_t
      buffers_capacity;  //Number of elements that can be stored in the MS buffers. TODO In future implementations this could be splited up, taking each buffer capacity in a different parameter.

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// ASNetwork Configuration Parameters
//--------------------------------------------------------------------
class ASNetworkConfig {
 public:
  ReduceNetwork_t reduce_network_type;  //Type of the ReduceNetwork configured in this moment
  std::size_t accumulation_buffer_enabled;
  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// ASwitch Configuration Parameters
//--------------------------------------------------------------------
class ASwitchConfig {
 public:
  std::size_t buffers_capacity;
  std::size_t input_ports;       //Number of input ports of the ASwitch. By  default in MAERI this is just 2
  std::size_t output_ports;      //Number of output ports of the ASwitch. By default  in MAERI this is 1
  std::size_t forwarding_ports;  //Nuber of forwarding ports of the ASwitch.
  std::size_t port_width;        //Bit width
  cycles_t latency;              //Latency of the AS to perform. This number is expressed in number of cycles. //TODO To implement

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// LookUpTable Configuration Parameters
//--------------------------------------------------------------------
class LookUpTableConfig {
 public:
  cycles_t latency;  //Latency of the LookUpTable to perform. This number must be expressed in number of cycles. 0 no supported
  std::size_t port_width;

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

//--------------------------------------------------------------------
// SDMemory Controller Configuration Parameters
//--------------------------------------------------------------------
class SDMemoryConfig {
 public:
  MemoryController_t mem_controller_type;       //Memory controller type (e.g., DENSE_WORKLOAD or SPARSE_GEMM)
  std::size_t write_buffer_capacity;            //Capacity of the buffers expressed in terms of number of elements
  std::size_t n_read_ports;                     //dn_bw
  std::size_t n_write_ports;                    //rn_bw
  std::size_t port_width;                       //Bit width
  std::size_t n_multiplier_configurations;      //Number of multiplier configurations
  std::size_t n_reduce_network_configurations;  //Number of reduce network configurations
  uint64_t weight_address;                      //Address where the weights are first stored
  uint64_t input_address;                       //Address ehere the inputs are first stored
  uint64_t output_address;                      //Address where the output are first stored
  uint32_t data_width;                          //Number of bytes allocated to each data element.
  uint32_t n_write_mshr;                        // Number of parallel write requests that can be done in parallel

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
  std::size_t print_stats_enabled;  //Specified whether the statistics must be printed.

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

  //Constructor runs reset()
  Config();

  //Load parameters from configuration file using TOML Syntax
  void loadFile(std::string config_file);

  //Reset parameters with default values
  void reset();

  //print the configuration parameters
  void printConfiguration(std::ofstream& out, std::size_t indent);

  //Indicates whether according to the hardware parameters, sparsity is enabled in the architecture
  bool sparsitySupportEnabled();

  //Indicates whether according to the hardware parameters, the operation of CONV itself is supported. Otherwise, the operation can be done
  //using GEMM operation
  bool convOperationSupported();
};

#endif
