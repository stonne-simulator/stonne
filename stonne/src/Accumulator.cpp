//Created 19/06/2019 by Francisco Munoz-Martinez

#include "Accumulator.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include "DataPackage.h"
#include "utility.h"

using namespace std;

/* This class represents an accumulator module. */

Accumulator::Accumulator(id_t id, std::string name, Config stonne_cfg, unsigned int n_accumulator)  : Unit(id, name) {
    this->n_accumulator = n_accumulator;
    this->input_ports = stonne_cfg.m_ASwitchCfg.input_ports;
    this->output_ports = stonne_cfg.m_ASwitchCfg.output_ports;
    //Collecting parameters from the configuration file
    this->buffers_capacity = stonne_cfg.m_ASwitchCfg.buffers_capacity;
    this->port_width = stonne_cfg.m_ASwitchCfg.port_width;
    this->latency = stonne_cfg.m_ASwitchCfg.latency;
    //End collecting parameters from the configuration file
    this->current_capacity = 0;
    this->inputConnection = NULL;
    this->outputConnection = NULL;
    this->input_fifo = new Fifo(this->buffers_capacity);
    this->output_fifo = new Fifo(this->buffers_capacity);
    this->local_cycle=0;
    this->current_psum=0;
    this->n_psums=0;
    this->operation_mode=ADDER;
   
}


Accumulator::~Accumulator() {
    delete this->input_fifo;
    delete this->output_fifo;
}

void Accumulator::setNPSums(unsigned int n_psums) {
	this->n_psums=n_psums;
	this->accumulatorStats.n_configurations++; //To track the stats
}


void Accumulator::resetSignals() {
    this->current_psum=0;
    this->operation_mode=ADDER;
    this->n_psums=0;

}

//Connection setters

void Accumulator::setInputConnection(Connection* inputConnection) {
    this->inputConnection = inputConnection;
}


void Accumulator::setOutputConnection(Connection* outputConnection) {
    this->outputConnection = outputConnection;
}


//Configuration settings (control signals)

void Accumulator::send() {
        if(!output_fifo->isEmpty()) {
            std::vector<DataPackage*> vector_to_send_parent;
            while(!output_fifo->isEmpty()) {
                 DataPackage* pck = output_fifo->pop();
                 vector_to_send_parent.push_back(pck);
            }

        //Sending if there is something
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ACCUMULATOR_FUNC] Cycle " << local_cycle << ", Accumulator " << this->n_accumulator << " has sent a psum to memory (FORWARDING DATA)" << std::endl;
#endif

	    this->accumulatorStats.n_memory_send++;
            this->outputConnection->send(vector_to_send_parent); //Send the data to the corresponding output
        }
    
}

void Accumulator::receive() { 
    if(this->inputConnection->existPendingData()) { //If there is data to receive on the left
    	std::vector<DataPackage*> data_received = this->inputConnection->receive(); //Copying the data to receive
	this->accumulatorStats.n_receives++;    //To track the stats
        for(int i=0; i<data_received.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ACCUMULATOR_FUNC] Cycle " << local_cycle << ", Accumulator " << this->n_accumulator << " has received a psum" << std::endl;
#endif
            input_fifo->push(data_received[i]); //Inserting to the local queuqe from connection
        }
    }

    return;
}

//Perform operation based on the parameter this->operation_mode
DataPackage* Accumulator::perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right) {
    //Extracting the values
    assert(pck_left->get_vn() == pck_right->get_vn()); // vn must fit
    
    data_t result; // Result of the operation
    switch(this->operation_mode) {
        case ADDER: //SUM
            result = pck_left->get_data() +  pck_right->get_data();
	    this->accumulatorStats.n_adds++;      //To track the stats
            //this->aswitchStats.n_2_1_sums++;  //Track the information
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ACCUMULATOR] Cycle " << local_cycle << ", Accumulator " << this->n_accumulator << " has performed an accumulation operation" << std::endl;
#endif

            break;
        default:
            assert(false); // This case must not occur in this type of configuration adder
    }
    //Creating the result package with the output
    DataPackage* result_pck = new DataPackage (sizeof(data_t), result, PSUM, 0, pck_left->get_vn(), this->operation_mode);  //TODO the size of the package corresponds with the data size
    //Adding to the creation list to be deleted afterward
//    this->psums_created.push_back(result_pck);
    return result_pck;
     

}

void Accumulator::route() {
    DataPackage* pck_received;
    if(!input_fifo->isEmpty()) {
        pck_received = input_fifo->pop();
        DataPackage* result;
        if(current_psum == 0) {  //There is no package yet to sum in this iteration
            //Creating package 0
            this->temporal_register = pck_received;
	    this->accumulatorStats.n_register_writes++;   //To track the stats
                 
        }
        else {
            result = perform_operation_2_operands(this->temporal_register, pck_received);
	    this->accumulatorStats.n_register_reads++; //To track the stats
            delete this->temporal_register;
            delete pck_received;
            this->temporal_register = result;
	    this->accumulatorStats.n_register_writes++;  //To track the stats
        }

        if(this->current_psum == (this->n_psums-1)) {
            this->output_fifo->push(this->temporal_register);
            this->current_psum = 0;
            
        }
        else {
            this->current_psum++;
        }

    }
        
}

//TODO
void Accumulator::cycle() {
    this->local_cycle+=1; 
    this->accumulatorStats.total_cycles++;  //To track the stats
    this->receive(); //Receive input
    this->route();
    this->send(); //Send towards the memory

}
/*
//Print the configuration of the Accumulator
void FEASwitch::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    out << ind(indent+IND_SIZE) << "\"Configuration\" : \"" <<get_string_adder_configuration(this->config_mode) << "\"" << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Augmented_link_enabled\" : " << this->fw_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Augmented_link_direction\" : \"" << get_string_fwlink_direction(this->fl_direction) << "\"" << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Left_child_enabled\" : " << this->left_child_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Right_child_enabled\" : " << this->right_child_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"BusID\" : " << this->busID << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"InputID\" : " << this->inputID << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Send_result_to_fold_node\" : " << this->forward_to_fold_node << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Send_result_to_memory\" : " << this->forward_to_memory  << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}
*/
//Print the statistics obtained during the execution

void Accumulator::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    this->accumulatorStats.print(out, indent+IND_SIZE);
    //Printing Fifos

    out << ind(indent+IND_SIZE) << ",\"InputFifo\" : {" << std::endl;
        this->input_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"OutputFifo\" : {" << std::endl;
        this->output_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}" << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent) << "}"; //TODO put ID

}

void Accumulator::printEnergy(std::ofstream& out, unsigned int indent){
    /* 
     This component prints:
         - Number of accumulator reads
         - Number of accumulator writes
	 - Number of sums performed by the accumulator
         - Number of reads and writes to the next fifos:
             * input_fifo: fifo to receive data
             * output_fifo: fifo to send data to memory
    */

    out << ind(indent) << "ACCUMULATOR READ=" << this->accumulatorStats.n_register_reads;
    out << ind(indent) << " WRITE=" << this->accumulatorStats.n_register_writes;
    out << ind(indent) << " ADD=" << this->accumulatorStats.n_adds << std::endl;

    //Fifos
    this->input_fifo->printEnergy(out, indent);
    this->output_fifo->printEnergy(out, indent);

}

