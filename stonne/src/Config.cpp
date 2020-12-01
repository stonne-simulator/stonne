//Created on 22/10/2019 by Francisco Munoz Martinez

#include "Config.h"
#include <iostream>
#include "types.h"
#include "utility.h"
#include "cpptoml.h"


Config::Config() {
    this->reset();
}

void Config::loadFile(std::string config_file) {
    auto config = cpptoml::parse_file(config_file);
    //General parameters
    auto print_stats_enabled_conf = config->get_as<bool>("print_stats_enabled");  //print_stats_enabled
    if(print_stats_enabled_conf) {
        this->print_stats_enabled = *print_stats_enabled_conf;
    }

    //DSNetwork Configuration Parameters
    auto n_switches_traversed_by_cycle_conf = config->get_qualified_as<unsigned int>("DSNetwork.n_switches_traversed_by_cycle"); //n_switches_traversed_by_cycle
    if(n_switches_traversed_by_cycle_conf) {
        this->m_DSNetworkCfg.n_switches_traversed_by_cycle = *n_switches_traversed_by_cycle_conf;
    }

    //DSwitch Configuration Parameters
    auto dswitch_latency_conf = config->get_qualified_as<unsigned int>("DSwitch.latency"); //Latency
    if(dswitch_latency_conf) {
        this->m_DSwitchCfg.latency = *dswitch_latency_conf;
    }

    auto dswitch_input_ports_conf = config->get_qualified_as<unsigned int>("DSwitch.input_ports");  //input_ports
    if(dswitch_input_ports_conf) {
        this->m_DSwitchCfg.input_ports = *dswitch_input_ports_conf;
    }

    auto dswitch_output_ports_conf = config->get_qualified_as<unsigned int>("DSwitch.output_ports");  //output_ports
    if(dswitch_output_ports_conf) {
        this->m_DSwitchCfg.output_ports = *dswitch_output_ports_conf;
    }

    auto dswitch_port_width_conf = config->get_qualified_as<unsigned int>("DSwitch.port_width");  //port_width
    if(dswitch_port_width_conf) {
        this->m_DSwitchCfg.port_width = *dswitch_port_width_conf;
    }

    //MSNetwork Configuration Parameters
    auto ms_size_conf = config->get_qualified_as<unsigned int>("MSNetwork.ms_size");
    if(ms_size_conf) {
        this->m_MSNetworkCfg.ms_size=*ms_size_conf;
    }

    auto ms_rows_conf = config->get_qualified_as<unsigned int>("MSNetwork.ms_rows");
    if(ms_rows_conf) {
        this->m_MSNetworkCfg.ms_rows=*ms_rows_conf;
    }

    auto ms_cols_conf = config->get_qualified_as<unsigned int>("MSNetwork.ms_cols");
    if(ms_cols_conf) {
        this->m_MSNetworkCfg.ms_cols=*ms_cols_conf;
    }

    auto multiplier_network_type_conf = config->get_qualified_as<std::string>("MSNetwork.type");  //Buffers_capacity
    if(multiplier_network_type_conf) {
        this->m_MSNetworkCfg.multiplier_network_type = get_type_multiplier_network_type(*multiplier_network_type_conf);
    }



    //MSwitch Configuration parameters
    auto mswitch_latency_conf = config->get_qualified_as<unsigned int>("MSwitch.latency"); //latency
    if(mswitch_latency_conf) {
        this->m_MSwitchCfg.latency = *mswitch_latency_conf; 
    }

    auto mswitch_input_ports_conf = config->get_qualified_as<unsigned int>("MSwitch.input_ports"); //input_ports
    if(mswitch_input_ports_conf) {
        this->m_MSwitchCfg.input_ports = *mswitch_input_ports_conf;
    }

    auto mswitch_output_ports_conf = config->get_qualified_as<unsigned int>("MSwitch.output_ports"); //output_ports
    if(mswitch_output_ports_conf) {
        this->m_MSwitchCfg.output_ports = *mswitch_output_ports_conf;
    }

    auto mswitch_forwarding_ports_conf = config->get_qualified_as<unsigned int>("MSwitch.forwarding_ports"); //forwarding_ports
    if(mswitch_forwarding_ports_conf) {
        this->m_MSwitchCfg.forwarding_ports = *mswitch_forwarding_ports_conf;
    }

    auto mswitch_port_width_conf = config->get_qualified_as<unsigned int>("MSwitch.port_width"); //port_width
    if(mswitch_port_width_conf) {
        this->m_MSwitchCfg.port_width = *mswitch_port_width_conf;
    }

    auto mswitch_buffers_capacity_conf = config->get_qualified_as<unsigned int>("MSwitch.buffers_capacity"); //buffers_capacity
    if(mswitch_buffers_capacity_conf) {
        this->m_MSwitchCfg.buffers_capacity = *mswitch_buffers_capacity_conf;
    }

    //ReduceNetwork Configuration Parameters
    auto reduce_network_type_conf = config->get_qualified_as<std::string>("ReduceNetwork.type");  //Buffers_capacity
    if(reduce_network_type_conf) {
        this->m_ASNetworkCfg.reduce_network_type = get_type_reduce_network_type(*reduce_network_type_conf);
    }

    auto accumulation_buffer_enabled_conf = config->get_qualified_as<unsigned int>("ReduceNetwork.accumulation_buffer_enabled");
    if(accumulation_buffer_enabled_conf) {
        this->m_ASNetworkCfg.accumulation_buffer_enabled = *accumulation_buffer_enabled_conf;
    }
    

    //ASwitch Configuration Parameters
    auto aswitch_buffers_capacity_conf = config->get_qualified_as<unsigned int>("ASwitch.buffers_capacity");  //Buffers_capacity
    if(aswitch_buffers_capacity_conf) {
        this->m_ASwitchCfg.buffers_capacity = *aswitch_buffers_capacity_conf;
    } 

    auto aswitch_input_ports_conf = config->get_qualified_as<unsigned int>("ASwitch.input_ports");    //input ports
    if(aswitch_input_ports_conf) {
        this->m_ASwitchCfg.input_ports = *aswitch_input_ports_conf;
    }

    auto aswitch_output_ports_conf = config->get_qualified_as<unsigned int>("ASwitch.output_ports");  //output ports
    if(aswitch_output_ports_conf) {
        this->m_ASwitchCfg.output_ports = *aswitch_output_ports_conf;
    }

    auto aswitch_forwarding_ports_conf = config->get_qualified_as<unsigned int>("ASwitch.forwarding_ports");  //forwarding ports
    if(aswitch_forwarding_ports_conf) {
        this->m_ASwitchCfg.forwarding_ports = *aswitch_forwarding_ports_conf;
    }

    auto aswitch_port_width_conf = config->get_qualified_as<unsigned int>("ASwitch.port_width");  //port width
    if(aswitch_port_width_conf) {
        this->m_ASwitchCfg.port_width = *aswitch_port_width_conf;
    }

    auto aswitch_latency_conf = config->get_qualified_as<unsigned int>("ASwitch.latency");  //latency
    if(aswitch_latency_conf) {
        this->m_ASwitchCfg.latency = *aswitch_latency_conf;
    }

    //LookupTable Configuration Parameters
    auto lookuptable_latency_conf = config->get_qualified_as<unsigned int>("LookUpTable.latency"); //latency
    if(lookuptable_latency_conf) {
        this->m_LookUpTableCfg.latency = *lookuptable_latency_conf;
    }

    auto lookuptable_port_width_conf = config->get_qualified_as<unsigned int>("LookUpTable.port_width"); //port_width
    if(lookuptable_port_width_conf) {
        this->m_LookUpTableCfg.port_width = *lookuptable_port_width_conf;
    }

    //SDMemory Configuration Parameters
    auto sdmemory_dn_bw_conf = config->get_qualified_as<unsigned int>("SDMemory.dn_bw");  //DN_BW
    if(sdmemory_dn_bw_conf) {
        this->m_SDMemoryCfg.n_read_ports = *sdmemory_dn_bw_conf;   
    }

    auto sdmemory_rn_bw_conf = config->get_qualified_as<unsigned int>("SDMemory.rn_bw");  //RN_BW
    if(sdmemory_rn_bw_conf) {
        this->m_SDMemoryCfg.n_write_ports = *sdmemory_rn_bw_conf; 
    }

    auto sdmemory_port_width_conf = config->get_qualified_as<unsigned int>("SDMemory.port_width");
    if(sdmemory_port_width_conf) {
        this->m_SDMemoryCfg.port_width = *sdmemory_port_width_conf;
    }

    auto memory_controller_type_conf = config->get_qualified_as<std::string>("SDMemory.controller_type");  
    if(memory_controller_type_conf) {
        this->m_SDMemoryCfg.mem_controller_type = get_type_memory_controller_type(*memory_controller_type_conf);
    }


   
}

void Config::reset() {

//General parameters
    print_stats_enabled=1;

// ---------------------------------------------------------
// DSNetwork Configuration Parameters
// ---------------------------------------------------------
    m_DSNetworkCfg.n_switches_traversed_by_cycle=23; //From paper. This is not implemented yet anyway.

// ---------------------------------------------------------
// DSwitch Configuration Parameters
// ---------------------------------------------------------

    //There is nothing yet
    m_DSwitchCfg.latency = 1; //Actually is less than 1. We do not implement this either
    m_DSwitchCfg.input_ports = 1;
    m_DSwitchCfg.output_ports=2;
    m_DSwitchCfg.port_width=16;  //Size in bits

// ---------------------------------------------------------
// MSNetwork Configuration Parameters
// ---------------------------------------------------------
    m_MSNetworkCfg.multiplier_network_type=LINEAR;
    m_MSNetworkCfg.ms_size=64;
    m_MSNetworkCfg.ms_rows=0; //Not initialized
    m_MSNetworkCfg.ms_cols=0; //Not initalized

// ---------------------------------------------------------
// MSwitch Configuration Parameters
// ---------------------------------------------------------
    m_MSwitchCfg.latency=1; //Latency in ns not implemented
    m_MSwitchCfg.input_ports=1;
    m_MSwitchCfg.output_ports=1;
    m_MSwitchCfg.forwarding_ports=1;
    m_MSwitchCfg.port_width=16;
    m_MSwitchCfg.buffers_capacity=2048;


// ---------------------------------------------------------
// ASNetwork Configuration Parameters
// ---------------------------------------------------------
    m_ASNetworkCfg.reduce_network_type=ASNETWORK;
    m_ASNetworkCfg.accumulation_buffer_enabled=0;

// ---------------------------------------------------------
// ASwitch Configuration Parameters
// ---------------------------------------------------------
    m_ASwitchCfg.buffers_capacity=256;
    m_ASwitchCfg.input_ports=2;
    m_ASwitchCfg.output_ports=1;
    m_ASwitchCfg.forwarding_ports=1;
    m_ASwitchCfg.port_width=16;
    m_ASwitchCfg.latency=1;

// ---------------------------------------------------------
// LookUpTable Configuration Parameters
// ---------------------------------------------------------
    m_LookUpTableCfg.latency=1; //Latency in ns not implemented yet
    m_LookUpTableCfg.port_width=1;

// ---------------------------------------------------------
// SDMemory Controller Configuration Parameters
// ---------------------------------------------------------
    m_SDMemoryCfg.mem_controller_type=MAERI_DENSE_WORKLOAD;
    m_SDMemoryCfg.write_buffer_capacity=256;
    m_SDMemoryCfg.n_read_ports=4; 
    m_SDMemoryCfg.n_write_ports=4; 
    m_SDMemoryCfg.port_width=16;


}

bool Config::sparsitySupportEnabled() {
    return m_SDMemoryCfg.mem_controller_type==SIGMA_SPARSE_GEMM; //If the controller is sparse, then sparsity is allowed
}
bool Config::convOperationSupported() {
    return m_SDMemoryCfg.mem_controller_type==MAERI_DENSE_WORKLOAD;
}


/** PRINTING FUNCTIONS **/

// -----------------------------------------------------------------------------------------------
// Config printing function
// -----------------------------------------------------------------------------------------------
void Config::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"hardwareConfiguration\" : {" << std::endl;
        //Printing general parameters
        out << ind(indent+IND_SIZE) << "\"print_stats_enabled\" : " << this->print_stats_enabled << "," << std::endl;  
        //Printing specific parameters of each unit
        this->m_DSNetworkCfg.printConfiguration(out, indent+IND_SIZE); 
        out << "," << std::endl;
        this->m_DSwitchCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_MSNetworkCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_MSwitchCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_ASNetworkCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_ASwitchCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_LookUpTableCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_SDMemoryCfg.printConfiguration(out, indent+IND_SIZE);
        out  << std::endl; //Take care of the comma since this is the last one
    out << ind(indent) << "}";
}


// -----------------------------------------------------------------------------------------------
// DSNetworkConfig printing function
// -----------------------------------------------------------------------------------------------
void DSNetworkConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"DSNetwork\" : {" << std::endl;
        //Printing DSNetwork configuration
        out << ind(indent+IND_SIZE) << "\"n_switches_traversed_by_cycle\" : " << this->n_switches_traversed_by_cycle << std::endl;
    out << ind(indent) << "}";
}


// -----------------------------------------------------------------------------------------------
// DSwitchConfig printing function
// -----------------------------------------------------------------------------------------------
void DSwitchConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"DSwitch\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl; 
        out << ind(indent+IND_SIZE) << "\"input_ports\" : " << this->input_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"output_ports\" : " << this->output_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width  << std::endl;
    out << ind(indent) << "}";
}


// -----------------------------------------------------------------------------------------------
// MSNetworkConfig printing function
// -----------------------------------------------------------------------------------------------
void MSNetworkConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"MSNetwork\" : {" << std::endl;
    out << ind(indent+IND_SIZE) << "\"multiplier_network_type\" : " << "\"" << get_string_multiplier_network_type(this->multiplier_network_type) << "\"" << ","  << std::endl;
    out << ind(indent+IND_SIZE) << "\"ms_rows\" : " << this->ms_rows << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"ms_columns\" : " << this->ms_cols << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"ms_size\" : " << this->ms_size  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// MSwitchConfig printing function
// -----------------------------------------------------------------------------------------------
void MSwitchConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"MSwitch\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"input_ports\" : " << this->input_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"output_ports\" : " << this->output_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"forwarding_ports\" : " << this->forwarding_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"buffers_capacity\" : " << this->buffers_capacity  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// ASNetworkConfig printing function
// -----------------------------------------------------------------------------------------------
void ASNetworkConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"ReduceNetwork\" : {" << std::endl;
    out << ind(indent+IND_SIZE) << "\"reduce_network_type\" : " << "\"" << get_string_reduce_network_type(this->reduce_network_type) << "\"" << ","  << std::endl;
    out << ind(indent+IND_SIZE) << "\"accumulation_buffer_enabled\" : " << this->accumulation_buffer_enabled  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// ASwitchConfig printing function
// -----------------------------------------------------------------------------------------------
void ASwitchConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"ASwitch\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"input_ports\" : " << this->input_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"output_ports\" : " << this->output_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"forwarding_ports\" : " << this->forwarding_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"buffers_capacity\" : " << this->buffers_capacity  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// LookUpTaleConfig printing function
// -----------------------------------------------------------------------------------------------
void LookUpTableConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"LookUpTable\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// SDMemoryConfig printing function
// -----------------------------------------------------------------------------------------------
void SDMemoryConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"SDMemory\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"mem_controller_type\" : " << "\"" << get_string_memory_controller_type(this->mem_controller_type) << "\""  << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"write_buffers_capacity\" : " << this->write_buffer_capacity << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"dn_bw\" : " << this->n_read_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"rn_bw\" : " << this->n_write_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width << std::endl;
    out << ind(indent) << "}";

}
