//Created by Francisco Munoz-Martinez on 13/06/2019

#include "MultiplierOS.h"
#include <assert.h>
#include "utility.h"
/*
*/

MultiplierOS::MultiplierOS(id_t id, std::string name, int row_num, int col_num,  Config stonne_cfg): 
Unit(id, name){
    this->row_num = row_num;
    this->col_num = col_num;
    //Extracting parameters from configuration file 
    this->latency = stonne_cfg.m_MSwitchCfg.latency;
    this->input_ports = stonne_cfg.m_MSwitchCfg.input_ports;
    this->output_ports = stonne_cfg.m_MSwitchCfg.output_ports;
    this->forwarding_ports = stonne_cfg.m_MSwitchCfg.forwarding_ports;
    this->buffers_capacity = stonne_cfg.m_MSwitchCfg.buffers_capacity;
    this->port_width = stonne_cfg.m_MSwitchCfg.port_width;
    this->ms_rows = stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ms_cols = stonne_cfg.m_MSNetworkCfg.ms_cols;
    //End of extracting parameters from the configuration file
 
    //Parameters initialization
    this->top_fifo = new Fifo(buffers_capacity);
    this->bottom_fifo = new Fifo(buffers_capacity);
    this->right_fifo = new Fifo(buffers_capacity);
    this->left_fifo = new Fifo(buffers_capacity);
    this->accbuffer_fifo = new Fifo(buffers_capacity);
    //End parameters initialization    


    this->local_cycle=0;
    this->forward_right=false; //Based on rows (windows) left and dimensions
    this->forward_bottom=false; //Based on columns (filters) left and dimensions
    this->VN = 0;
    this->num = this->row_num * this->ms_cols + this->col_num; //Multiplier ID, used just for information
    
}

MultiplierOS::MultiplierOS(id_t id, std::string name, int row_num, int col_num,  Config stonne_cfg, Connection* left_connection, Connection* right_connection, Connection* top_connection, Connection* bottom_connection) : MultiplierOS(id, name, row_num, col_num, stonne_cfg) {
    this->setLeftConnection(left_connection);
    this->setRightConnection(right_connection);
    this->setTopConnection(top_connection);
    this->setBottomConnection(bottom_connection);

}

MultiplierOS::~MultiplierOS() {
    delete this->left_fifo;
    delete this->right_fifo;
    delete this->top_fifo;
    delete this->bottom_fifo;
}

void MultiplierOS::resetSignals() {

    this->forward_right=false; 
    this->forward_bottom=false;
    this->VN=0;
    while(!left_fifo->isEmpty()) {
        delete left_fifo->pop();
    }

    while(!right_fifo->isEmpty()) {
        delete right_fifo->pop();
    }

    while(!top_fifo->isEmpty()) {
        delete top_fifo->pop();
    }

    while(!bottom_fifo->isEmpty()) {
        delete bottom_fifo->pop();
    }


}
void MultiplierOS::setLeftConnection(Connection* left_connection) { 
    this->left_connection = left_connection;
}
void MultiplierOS::setRightConnection(Connection* right_connection) { //Set the right connection of the ms
    this->right_connection = right_connection;
}

void MultiplierOS::setTopConnection(Connection* top_connection) {
    this->top_connection = top_connection;
}

void MultiplierOS::setBottomConnection(Connection* bottom_connection) { //Set the input connection of the switch
    this->bottom_connection = bottom_connection;
}

void MultiplierOS::setAccBufferConnection(Connection* accbuffer_connection) { //Set the output connection 
    this->accbuffer_connection = accbuffer_connection;
}


void MultiplierOS::configureBottomSignal(bool bottom_signal) {
    this->forward_bottom = bottom_signal;
}

void MultiplierOS::configureRightSignal(bool right_signal) {
    this->forward_right = right_signal;
}

void MultiplierOS::setVirtualNeuron(unsigned int VN) {
    this->VN = VN;
    this->mswitchStats.n_configurations++;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MultiplierOS "  << ". VirtualNeuron: " << this->VN << std::endl;
#endif

}


void MultiplierOS::send() { //Send the result through the outputConnection
	//Sending weights
        std::vector<DataPackage*> vector_to_send_weights;
	while(!bottom_fifo->isEmpty()) { 
            DataPackage* data_to_send = bottom_fifo->pop();
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MultiplierOS " << this->num << " has forward a data to the bottom connection" << std::endl;
#endif
            vector_to_send_weights.push_back(data_to_send);
        }
	if(vector_to_send_weights.size() > 0) {
	    this->mswitchStats.n_bottom_forwardings_send++;
            this->bottom_connection->send(vector_to_send_weights);
	}

	//Sending activations
	std::vector<DataPackage*> vector_to_send_activations;
        while(!right_fifo->isEmpty()) { 
            DataPackage* data_to_send = right_fifo->pop();
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MultiplierOS " << this->num << " has forward a data to the right connection" << std::endl;
#endif
            vector_to_send_activations.push_back(data_to_send);
        }
	if(vector_to_send_activations.size() > 0) {
	    this->mswitchStats.n_right_forwardings_send++;
            this->right_connection->send(vector_to_send_activations);
	}

	//Sending to the accumulation buffer. Note that this might be done inside the PE, 
	//however since we already have an accumulation buffer, we will use this and will remove
	//the connection between them. 
	
        //Sending psums to the accumulation buffer
        std::vector<DataPackage*> vector_to_send_psums;
        while(!accbuffer_fifo->isEmpty()) {  
            DataPackage* data_to_send = accbuffer_fifo->pop();
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MultiplierOS " << this->num << " has forward a data to the accumulation buffer" << std::endl;
#endif
            vector_to_send_psums.push_back(data_to_send);
        }
	if(vector_to_send_psums.size() > 0) {
            this->accbuffer_connection->send(vector_to_send_psums); 
	}




}


void MultiplierOS::receive() {  //Receive a package either from the left or the top connection

    if(left_connection->existPendingData()) {
        std::vector<DataPackage*> data_received  = left_connection->receive(); 
        assert(data_received.size() == 1);
        for(int i=0; i<data_received.size(); i++) {
            DataPackage* pck = data_received[i];
            assert(pck->get_data_type()==IACTIVATION);
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MultiplierOS " << this->num << " has received an IACTIVATION from the left connection" << std::endl;
#endif
	    this->mswitchStats.n_left_forwardings_receive++;


            left_fifo->push(pck);
        }
    
    }

    if(top_connection->existPendingData()) {
        std::vector<DataPackage*> data_received  = top_connection->receive();     
        assert(data_received.size() == 1);
        for(int i=0; i<data_received.size(); i++) {
            DataPackage* pck = data_received[i];
            assert(pck->get_data_type()==WEIGHT);
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MultiplierOS " << this->num << " has received a weight from the top connection" << std::endl;
#endif
	    this->mswitchStats.n_top_forwardings_receive++;

            top_fifo->push(pck);
        }

    
    }
    return;
}

//Perform multiplication with the weight and the activation
DataPackage* MultiplierOS::perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right) {
    //Extracting the values
    data_t result; // Result of the operation
    result = pck_left->get_data() *  pck_right->get_data(); 
#ifdef DEBUG_MSWITCH_FUNC
    std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MultiplierOS " << this->num << " has performed a multiplication" << std::endl;
#endif

    
    //Creating the result package with the output
    DataPackage* result_pck = new DataPackage (sizeof(data_t), result, PSUM, this->num, this->VN, MULTIPLIER);  //TODO the size of the package corresponds with the data size
    //Adding to the creation list to be deleted afterward
    //this->psums_created.push_back(result_pck);
    this->mswitchStats.n_multiplications++; // Track a multiplication
    return result_pck;


}

//Si es la primera iteracion no recibas del fw link
//Si es la ultima iteracion no envies al fw link
void MultiplierOS::cycle() { //Computing a cycle
//    std::cout << "MS: " << this->num << ". weight fifo size: " << weight_fifo->size() << " N_Folding: " << n_folding << std::endl;
    local_cycle+=1;
    this->mswitchStats.total_cycles++;
    this->receive(); //From top and left
    if((!left_fifo->isEmpty()) && (!top_fifo->isEmpty())) { //If both queues are not empty
            DataPackage* activation = left_fifo->pop(); //Get the activation and remove from fifo
            
            DataPackage* weight = top_fifo->pop(); //get the weight and remove from fifo
            DataPackage* pck_result = perform_operation_2_operands(activation, weight); //Creating the psum package
            accbuffer_fifo->push(pck_result); //Sending to the accbuffer to be accumulated in OS manner

	    //Forwarding the weight and the activation
	    if(forward_right) {  //If this ms is not in the last column (i.e., filter in conv)
	        right_fifo->push(activation);
	    }
	    else {
                delete activation;
	    } 
	    if(forward_bottom) { //if this ms is not in the last row (i.e., last conv window in conv))
	        bottom_fifo->push(weight);
	    }

	    else {
                delete weight;
	    }

	    this->send();
       
    } 

    
}

void MultiplierOS::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; 
    out << ind(indent+IND_SIZE) << "\"VN\" : " << this->VN  << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent respo    
}

void MultiplierOS::printStats(std::ofstream& out, unsigned int indent) {
    
    out << ind(indent) << "{" << std::endl; //TODO put ID
    this->mswitchStats.print(out, indent+IND_SIZE);
    //Printing Fifos 

    out << ind(indent+IND_SIZE) << ",\"TopFifo\" : {" << std::endl;
        this->top_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);         
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"LeftFifo\" : {" << std::endl;
        this->left_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"RightFifo\" : {" << std::endl;
        this->right_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl;; //Take care. Do not print endl here. This is parent responsability
   
    out << ind(indent+IND_SIZE) << "\"BottomFifo\" : {" << std::endl;
        this->bottom_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl;; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"OutputFifo\" : {" << std::endl;
        this->accbuffer_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}" << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent) << "}"; 




   


}

void MultiplierOS::printEnergy(std::ofstream& out, unsigned int indent) {
    /*
         This component prints:
             - MSwitch multiplications
             - MSwitch forwardings
             - Counters for the FIFOs (registers in TPU, ie., we will use the access count of a register):
                 * top_fifo: the fifo that is used to receive the weights 
                 * left_fifo: Fifo used to receive the activations
                 * right_fifo: Fifo used to send the activations
                 * bottom_fifo: FIFO used to send the weights
                 * output_fifo: FIFO used to store final result once accumulated
   */

    //Multiplier counters
    
    out << ind(indent) << "MULTIPLIER MULTIPLICATION=" << this->mswitchStats.n_multiplications;
    out << ind(indent) << " CONFIGURATION=" << this->mswitchStats.n_configurations << std::endl;

    //Fifo counters
    top_fifo->printEnergy(out, indent);
    left_fifo->printEnergy(out, indent);
    right_fifo->printEnergy(out, indent);
    bottom_fifo->printEnergy(out, indent);
    accbuffer_fifo->printEnergy(out, indent);
    
    
}


