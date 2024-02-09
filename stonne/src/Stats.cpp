#include "Stats.h"
#include "utility.h"


//--------------- ConnectionStats implementation ---------------
//--------------------------------------------------------------

ConnectionStats::ConnectionStats() {
    this->reset();
}

void ConnectionStats::reset() {
    this->n_sends=0;
    this->n_receives=0;
}

void ConnectionStats::print(std::ofstream& out, unsigned int indent) {
    
}


//--------------- FifoStats implementation ---------------
// -------------------------------------------------------
FifoStats::FifoStats() {
    this->reset();
}

void FifoStats::reset() {
    this->n_pops=0;
    this->n_pushes=0;
    this->n_fronts=0;
    this->max_occupancy=0;
}

void FifoStats::print(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"N_pops\" : " << this->n_pops << "," << std::endl;
    out << ind(indent) << "\"N_pushes\" : " << this->n_pushes << "," << std::endl;
    out << ind(indent) << "\"N_fronts\" : " << this->n_fronts << "," << std::endl;
    out << ind(indent) << "\"Max_occupancy\" : " << this->max_occupancy  << std::endl;
}

//--------------- ASwitchStats implementation ---------------
// ----------------------------------------------------------

DSwitchStats::DSwitchStats() {
    this->reset();
}

void DSwitchStats::reset() {
    this->total_cycles=0;
    this->n_broadcasts=0;
    this->n_unicasts=0;
    this->n_left_sends=0;
    this->n_right_sends=0;
}

void DSwitchStats::print(std::ofstream& out, unsigned int indent) {
    counter_t idle_cycles=this->total_cycles-(this->n_broadcasts+this->n_unicasts); //Calculated statistic
    out << ind(indent) << "\"Total_cycles\" : " << this->total_cycles << "," << std::endl;
    out << ind(indent) << "\"Idle_cycles_dswitch\" : " << idle_cycles << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_broadcasts\" : " << this->n_broadcasts << "," << std::endl;
    out << ind(indent) << "\"N_unicasts\" : " << this->n_unicasts << "," << std::endl;
    out << ind(indent) << "\"N_left_sends\" : " << this->n_left_sends << "," << std::endl;
    out << ind(indent) << "\"N_right_sends\" : " << this->n_right_sends  << std::endl; 
}


//--------------- MSwitchStats implementation ---------------
// ----------------------------------------------------------

MSwitchStats::MSwitchStats() {
    this->reset();
}

void MSwitchStats::reset() {
    this->total_cycles=0;
    this->n_multiplications=0;
    this->n_input_forwardings_send=0;
    this->n_input_forwardings_receive=0;
    this->n_inputs_receive=0;
    this->n_weights_receive=0;
    this->n_weight_fifo_flush=0;
    this->n_psums_receive=0;
    this->n_psum_forwarding_send=0;
    this->n_configurations=0;
}

void MSwitchStats::print(std::ofstream& out, unsigned int indent) {
    counter_t idle_cycles=this->total_cycles-(this->n_multiplications+this->n_psum_forwarding_send); //Calculated statistic
    out << ind(indent) << "\"Total_cycles\" : " << this->total_cycles << "," << std::endl;
    out << ind(indent) << "\"Idle_cycles_mswitch\" : " << idle_cycles << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_multiplications\" : " << this->n_multiplications << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_input_forwardings_send\" : " << this->n_input_forwardings_send << "," << std::endl;
    out << ind(indent) << "\"N_input_forwardings_receive\" : " << this->n_input_forwardings_receive << "," << std::endl;
    out << ind(indent) << "\"N_inputs_receive_from_memory\" : " << this->n_inputs_receive << "," << std::endl;
    out << ind(indent) << "\"N_weights_receive_from_memory\" : " << this->n_weights_receive << "," << std::endl;
    out << ind(indent) << "\"N_weight_fifo_flush\" : " << this->n_weight_fifo_flush << "," << std::endl;
    out << ind(indent) << "\"N_psums_receive\" : " << this->n_psums_receive << "," << std::endl;
    out << ind(indent) << "\"N_psum_forwarding_send\" : " << this->n_psum_forwarding_send  << "," << std::endl;
    out << ind(indent) << "\"N_configurations\" : " << this->n_configurations  << std::endl;  //Take care. We do not print the comma here.

}

//--------------- MultiplierOS stats implementation ---------------
// ----------------------------------------------------------

MultiplierOSStats::MultiplierOSStats() {
    this->reset();
}

void MultiplierOSStats::reset() {
    this->total_cycles=0;
    this->n_multiplications=0;
    this->n_bottom_forwardings_send=0;
    this->n_top_forwardings_receive=0;
    this->n_right_forwardings_send=0;             
    this->n_left_forwardings_receive=0;
    this->n_configurations=0;
}

void MultiplierOSStats::print(std::ofstream& out, unsigned int indent) {
    counter_t idle_cycles=this->total_cycles-(this->n_multiplications); //Calculated statistic
    out << ind(indent) << "\"Total_cycles\" : " << this->total_cycles << "," << std::endl;
    out << ind(indent) << "\"Idle_cycles_osmswitch\" : " << idle_cycles << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_multiplications\" : " << this->n_multiplications << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_bottom_forwardings_send\" : " << this->n_bottom_forwardings_send << "," << std::endl;
    out << ind(indent) << "\"N_top_forwardings_receive\" : " << this->n_top_forwardings_receive << "," << std::endl;
    out << ind(indent) << "\"N_right_forwardings_send\" : " << this->n_right_forwardings_send << "," << std::endl;
    out << ind(indent) << "\"N_left_forwardings_receive\" : " << this->n_left_forwardings_receive << "," << std::endl;
    out << ind(indent) << "\"N_configurations\" : " << this->n_configurations  << std::endl;  //Take care. We do not print the comma here.

}



//--------------- ASwitchStats implementation ---------------
// ----------------------------------------------------------

ASwitchStats::ASwitchStats() {
    this->reset();
}

void ASwitchStats::reset() {
    this->total_cycles=0;
    this->n_2_1_sums=0;
    this->n_2_1_comps=0;
    this->n_3_1_sums=0;
    this->n_3_1_comps=0;
    this->n_parent_send=0;
    this->n_augmented_link_send=0;
    this->n_memory_send=0;
    this->n_configurations=0;
}

void ASwitchStats::print(std::ofstream& out, unsigned int indent) {
    counter_t idle_cycles=this->total_cycles-(this->n_2_1_sums+this->n_2_1_comps+this->n_3_1_sums+this->n_3_1_comps); //Calculated statistic
    out << ind(indent) << "\"Total_cycles\" : " << this->total_cycles << "," << std::endl;
    out << ind(indent) << "\"Idle_cycles_aswitch\" : " << idle_cycles << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_2_1_sums\" : " << this->n_2_1_sums << "," << std::endl;
    out << ind(indent) << "\"N_2_1_comps\" : " << this->n_2_1_comps << "," << std::endl;
    out << ind(indent) << "\"N_3_1_sums\" : " << this->n_3_1_sums << "," << std::endl;
    out << ind(indent) << "\"N_3_1_comps\" : " << this->n_3_1_comps << "," << std::endl;
    out << ind(indent) << "\"N_parent_send\" : " << this->n_parent_send << "," << std::endl;
    out << ind(indent) << "\"N_augmentendLink_send\" : " << this->n_augmented_link_send << "," << std::endl;
    out << ind(indent) << "\"N_memory_send\" : " << this->n_memory_send  << "," << std::endl;
    out << ind(indent) << "\"N_configurations\" : " << this->n_configurations  << std::endl;  //Take care. We do not print the comma here. Last one

}

//--------------- AccumulatorStats implementation ---------------
// ----------------------------------------------------------

AccumulatorStats::AccumulatorStats() {
    this->reset();
}

void AccumulatorStats::reset() {
    this->total_cycles=0;
    this->n_adds=0;
    this->n_memory_send=0;
    this->n_receives=0;
    this->n_register_reads=0;
    this->n_register_writes=0;
    this->n_configurations=0;
}

void AccumulatorStats::print(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"Total_cycles\" : " << this->total_cycles << "," << std::endl;
    out << ind(indent) << "\"N_adds\" : " << this->n_adds << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_memory_send\" : " << this->n_memory_send << "," << std::endl;
    out << ind(indent) << "\"N_receives\" : " << this->n_receives << "," << std::endl;
    out << ind(indent) << "\"N_register_reads\" : " << this->n_register_reads << "," << std::endl;
    out << ind(indent) << "\"N_register_writes\" : " << this->n_register_writes << "," << std::endl;
    out << ind(indent) << "\"N_configurations\" : " << this->n_configurations  << std::endl;  //Take care. We do not print the comma here. Last one

}



//--------------- SDMemory implementation ---------------
// ------------------------------------------------------

SDMemoryStats::SDMemoryStats() {
    this->reset();
}

void SDMemoryStats::reset() {
    this->total_cycles=0;
    this->n_SRAM_weight_reads=0;
    this->n_SRAM_input_reads=0;
    this->n_SRAM_psum_reads=0;
    this->n_SRAM_psum_writes=0;
    this->n_DRAM_psum_writes=0;
    this->sta_sparsity=0;
    this->str_sparsity=0;
    this->dataflow=CNN_DATAFLOW;
    this->n_sta_vectors_at_once_avg = 0;
    this->n_sta_vectors_at_once_max = 0;
    this->n_reconfigurations=0;
    this->n_cycles_multiplying=0;
    this->n_cycles_merging=0;
}

void SDMemoryStats::print(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"Total_cycles\" : " << this->total_cycles << "," << std::endl;
    out << ind(indent) << "\"N_cycles_multiplying\" : " << this->n_cycles_multiplying << "," << std::endl;
    out << ind(indent) << "\"N_cycles_merging\" : " << this->n_cycles_merging << "," << std::endl;
    out << ind(indent) << "\"N_SRAM_weight_reads\" : " << this->n_SRAM_weight_reads << "," << std::endl; //calculated statistic
    out << ind(indent) << "\"N_SRAM_input_reads\" : " << this->n_SRAM_input_reads << "," << std::endl;
    out << ind(indent) << "\"N_SRAM_psum_reads\" : " << this->n_SRAM_psum_reads << "," << std::endl;
    out << ind(indent) << "\"N_SRAM_psum_writes\" : " << this->n_SRAM_psum_writes << ","  << std::endl;  
    out << ind(indent) << "\"N_DRAM_psum_writes\" : " << this->n_DRAM_psum_writes << ","  << std::endl;
    out << ind(indent) << "\"Dataflow\" : " << "\"" << get_string_dataflow_type(this->dataflow) << "\"" << ","  << std::endl;
    out << ind(indent) << "\"STA_sparsity\" : " << this->sta_sparsity << ","  << std::endl;
    out << ind(indent) << "\"STR_sparsity\" : " << this->str_sparsity << ","  << std::endl;
    out << ind(indent) << "\"STA_vectors_at_once_avg\" : " << this->n_sta_vectors_at_once_avg << ","  << std::endl;
    out << ind(indent) << "\"STA_vectors_at_once_max\" : " << this->n_sta_vectors_at_once_max << ","  << std::endl;

    out << ind(indent) << "\"N_reconfigurations\" : " << this->n_reconfigurations << ","  << std::endl;

  
    //Printing the arrays with the use of the read ports
    //n_sram_read_ports_weights_use
    out << ind(indent) << "\"N_SRAM_read_ports_weights_use\" : [" << std::endl;
    for(int i=0; i<n_SRAM_read_ports_weights_use.size(); i++) {
        out << ind(indent+IND_SIZE) << n_SRAM_read_ports_weights_use[i];
        if(i==(n_SRAM_read_ports_weights_use.size()-1)) { //If it is the last one, comma is not added
            out << std::endl;
        }

        else {
            out << "," << std::endl;
        }
    }
    out << ind(indent) << "]," << std::endl;

    //n_sram_read_ports_inputs_use
    out << ind(indent) << "\"N_SRAM_read_ports_inputs_use\" : ["   << std::endl;
    for(int i=0; i<n_SRAM_read_ports_inputs_use.size(); i++) {
        out << ind(indent+IND_SIZE) << n_SRAM_read_ports_inputs_use[i];
        if(i==(n_SRAM_read_ports_inputs_use.size()-1)) { //If it is the last one, comma is not added
            out << std::endl;
        }

        else {
            out << "," << std::endl;
        }

    }
    out << ind(indent) << "]," << std::endl;

    //n_sram_read_ports_psums_use
    out << ind(indent) << "\"N_SRAM_read_ports_psums_use\" : [" << std::endl;
    for(int i=0; i<n_SRAM_read_ports_psums_use.size(); i++) {
        out << ind(indent+IND_SIZE) << n_SRAM_read_ports_psums_use[i];
        if(i==(n_SRAM_read_ports_psums_use.size()-1)) { //If it is the last one, comma is not added
            out << std::endl;
        }

        else {
            out << "," << std::endl;
        }

    }
    out << ind(indent) << "]," << std::endl;  
   
        //n_sram_write_ports_use
    out << ind(indent) << "\"N_SRAM_write_ports_use\" : [" << std::endl;
    for(int i=0; i<n_SRAM_write_ports_use.size(); i++) {
        out << ind(indent+IND_SIZE) << n_SRAM_write_ports_use[i];
        if(i==(n_SRAM_write_ports_use.size()-1)) { //If it is the last one, comma is not added
            out << std::endl;
        }

        else {
            out << "," << std::endl;
        }

    }
    out << ind(indent) << "]" << std::endl;   //Take care of the comma here. This is the last array

}


//--------------- SDMemory implementation ---------------
// ------------------------------------------------------

CollectionBusLineStats::CollectionBusLineStats() {
    this->reset();
}

void CollectionBusLineStats::reset() {
    this->total_cycles=0;
    this->n_times_conflicts=0;
    this->n_conflicts_average=0;
    this->n_sends=0;
}

void CollectionBusLineStats::print(std::ofstream& out, unsigned int indent) {
    if(total_cycles >0) { //To make sure it does not break in this extreme case
        this->n_conflicts_average=(int)(this->n_conflicts_average/this->total_cycles); 
    }
    out << ind(indent) << "\"Total_cycles\" : " << this->total_cycles << "," << std::endl;
    out << ind(indent) << "\"N_Times_conflicts\" : " << this->n_times_conflicts << "," << std::endl;
    out << ind(indent) << "\"N_Conflicts_Average\" : " << this->n_conflicts_average << "," << std::endl;
    out << ind(indent) << "\"N_sends\" : " << this->n_sends << "," << std::endl;

    //Printing arrays (i.e., for each input (fifo))
    out << ind(indent) << "\"n_inputs_receive\" : [" << std::endl; 
    for(int i=0; i<n_inputs_receive.size(); i++) {
        out << ind(indent+IND_SIZE) << n_inputs_receive[i];
        if(i==(n_inputs_receive.size()-1)) { //If it is the last one, comma is not added
            out << std::endl;
        }

        else {
            out << "," << std::endl;
        }

    }
    out << ind(indent) << "]" << std::endl;   //Take care of the comma here. This is the last array

   
}
