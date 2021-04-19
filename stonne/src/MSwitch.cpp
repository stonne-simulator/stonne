//Created by Francisco Munoz-Martinez on 13/06/2019

#include "MSwitch.h"
#include <assert.h>
#include "utility.h"
/*
*/

MSwitch::MSwitch(id_t id, std::string name, int num, Config stonne_cfg) : 
Unit(id, name){
    this->num = num;
    //Extracting parameters from configuration file 
    this->latency = stonne_cfg.m_MSwitchCfg.latency;
    this->input_ports = stonne_cfg.m_MSwitchCfg.input_ports;
    this->output_ports = stonne_cfg.m_MSwitchCfg.output_ports;
    this->forwarding_ports = stonne_cfg.m_MSwitchCfg.forwarding_ports;
    this->buffers_capacity = stonne_cfg.m_MSwitchCfg.buffers_capacity;
    this->port_width = stonne_cfg.m_MSwitchCfg.port_width;
    //End of extracting parameters from the configuration file
 
    //Parameters initialization
    this->activation_fifo = new Fifo(buffers_capacity);
    this->forwarding_output_fifo = new Fifo(buffers_capacity);
    this->forwarding_input_fifo = new Fifo(buffers_capacity);
    this->weight_fifo = new Fifo(buffers_capacity);
    this->psum_fifo = new Fifo(buffers_capacity);
    //End parameters initialization    

    //Signals
    this->inputForwardingEnabled = false;
    this->outputForwardingEnabled = false;

    this->current_n_windows = 0;
    this->current_n_folding = 0;
    this->forward_psum = false;  //Indicates whether it has to forward the psum with the iterations controlled by the memory
    this->direct_forward_psum = false; //Indicates whether it has to forward the psum, ALWAYS.
    this->local_cycle=0;
    this->n_windows = 0;
    this->n_folding = 0;
    this->VN=-1; //Not configured 
    
}

MSwitch::MSwitch(id_t id, std::string name, int num, Config stonne_cfg, Connection* outputConnection, Connection* inputConnection) : MSwitch(id, name, num, stonne_cfg) {
    this->setOutputConnection(outputConnection);
    this->setInputConnection(inputConnection);

}

MSwitch::~MSwitch() {
    delete this->activation_fifo;
    delete this->weight_fifo;
    delete this->psum_fifo;
    delete this->forwarding_output_fifo;
    delete this->forwarding_input_fifo;
}

void MSwitch::resetSignals() {


    this->receive(inputConnection);
    //Signals
    this->inputForwardingEnabled = false;
    this->outputForwardingEnabled = false;

    //this->current_n_windows = 0;
    //this->current_n_folding = 0;
    this->forward_psum = false; 
    this->direct_forward_psum = false;
    //this->n_windows = 0;
    //this->n_folding = 0;

    this->VN=-1; //Not configured 
        //while(!weight_fifo->isEmpty()) {
        //weight_fifo->pop();
    //}
    //Ordering the filters again
    while(current_n_folding < n_folding) {
	if(!weight_fifo->isEmpty()) {
            DataPackage* weight = weight_fifo->pop(); //get the weight and then pushing again at the end of the fifo
            weight_fifo->push(weight);
	}
	current_n_folding++;
    }

    current_n_folding = 0;
    current_n_windows = 0;

    while(!activation_fifo->isEmpty()) {
        activation_fifo->pop();
    }

      while(!psum_fifo->isEmpty()) {
        psum_fifo->pop();
    }

      while(!forwarding_output_fifo->isEmpty()) {
        forwarding_output_fifo->pop();
    }

     while(!forwarding_input_fifo->isEmpty()) {
        forwarding_input_fifo->pop();
    }


}
void MSwitch::setOutputConnection(Connection* outputConnection) { //Set the left connection of the switch
    this->outputConnection = outputConnection;
}
void MSwitch::setInputForwardingConnection(Connection* inputForwardingConnection) { //Set the right connection of the switch
    this->inputForwardingConnection = inputForwardingConnection;
}

void MSwitch::setOutputForwardingConnection(Connection* outputForwardingConnection) {
    this->outputForwardingConnection = outputForwardingConnection;
}

void MSwitch::setInputConnection(Connection* inputConnection) { //Set the input connection of the switch
    this->inputConnection = inputConnection;
}

void MSwitch::setNWindows(unsigned int n_windows) {
    this->n_windows = n_windows;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MSwitch " << this->num << ". n_windows: " << this->n_windows << std::endl;
#endif
}

void MSwitch::setNFolding(unsigned int n_folding) {
    this->n_folding = n_folding;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MSwitch " << this->num << ". n_folding: " << this->n_folding << std::endl;
#endif
}

void MSwitch::setForwardPsum(bool forward_psum) {
    this->forward_psum = forward_psum;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MSwitch " << this->num << ". forward_psum: " << this->forward_psum << std::endl;
#endif

}

void MSwitch::setDirectForwardPsum(bool direct_forward_psum) {
    this->direct_forward_psum = direct_forward_psum;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MSwitch " << this->num << ". direct_forward_psum: " << this->direct_forward_psum << std::endl;
#endif

}


void MSwitch::send() { //Send the result through the outputConnection
        std::vector<DataPackage*> vector_to_send;
	while(!psum_fifo->isEmpty()) { //There must exist data in the switch //TODO use the ports number too
            //std::cout << "MSwitch " << this->num << " Computed at cycle " << this->local_cycle << std::endl;
            DataPackage* data_to_send = psum_fifo->pop();
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " has sent a psum to the parent" << std::endl;
#endif
            vector_to_send.push_back(data_to_send);
        }
        this->outputConnection->send(vector_to_send); //Send the result to the output towards the RN

}

//Send forwarding activation
void MSwitch::forward(DataPackage* activation_to_forward) {
    //There is no queue to store intermedium activations (see figure of the Mswitch in the MAERI paper)
    std::vector<DataPackage*> vector_to_send;
    vector_to_send.push_back(activation_to_forward);
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " has sent an IACTIVATION to the forwarding link" << std::endl;
#endif
    this->outputForwardingConnection->send(vector_to_send); //Sending the activation to the left        
    this->mswitchStats.n_input_forwardings_send++;
}


void MSwitch::setVirtualNeuron(unsigned int VN) {
    this->VN = VN; 
    this->mswitchStats.n_configurations++;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MSwitch " << this->num << ". VirtualNeuron: " << this->VN << std::endl;
#endif

}

void MSwitch::setInputForwardingEnabled(bool inputForwardingEnabled) {
    this->inputForwardingEnabled = inputForwardingEnabled;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MSwitch " << this->num << ". inputForwardingEnabled: " << this->inputForwardingEnabled << std::endl;
#endif

}

void MSwitch::setOutputForwardingEnabled(bool outputForwardingEnabled) {
    this->outputForwardingEnabled = outputForwardingEnabled;
#ifdef DEBUG_MSWITCH_CONFIG
    std::cout << "[MSWITCH_CONFIG] MSwitch " << this->num << ". outputForwardingEnabled: " << this->outputForwardingEnabled << std::endl;
#endif

}

void MSwitch::receive(Connection* connection) {  //Receive a package from the inputConnection or forwardingConnection 

    if(connection == inputForwardingConnection) {
        if(connection->existPendingData()) {
            std::vector<DataPackage*> data_received  = connection->receive(); //Copying the data to receive //TODO check the number of elements with ports
            this->mswitchStats.n_input_forwardings_receive++;  //Tracking the reception of an input
            assert(data_received.size() == 1);
            for(int i=0; i<data_received.size(); i++) {
                DataPackage* pck = data_received[i];
                assert(pck->get_data_type()==IACTIVATION);
#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " has received an IACTIVATION from the forwarding link" << std::endl;
#endif


                forwarding_input_fifo->push(pck);
            }
        }
    }
    if(connection->existPendingData()) { //If there is data to receive
        //TODO PROOF WITH INPUT. LATER DEPENDS ON CONFIGURATION        
        std::vector<DataPackage*> data_received  = connection->receive(); //Copying the data to receive //TODO check the number of elements with ports etc
        assert(data_received.size() == 1);
        for(int i=0; i<data_received.size(); i++) {
            //Check if is an activation or weight
            DataPackage* pck = data_received[i];
            if(pck->get_data_type()==IACTIVATION) {
                activation_fifo->push(pck);
                this->mswitchStats.n_inputs_receive++;  //Tracking the information

#ifdef DEBUG_MSWITCH_FUNC
            std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " has received an IACTIVATION from the distribution network" << std::endl;
#endif

            }

            else if(pck->get_data_type()==WEIGHT) {
#ifdef DEBUG_MSWITCH_FUNC
                std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " has received a WEIGHT from the distribution network" << std::endl;
#endif
                if(weight_fifo->size() == n_folding) { //If the fifo already has n_folding weights means that this weight is a new distirbution weight phase and therefore we clear the fifo
                    //Removing all the elements
#ifdef DEBUG_MSWITCH_FUNC
                std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " clears the weight fifo" << std::endl;
#endif
                    while(!weight_fifo->isEmpty()) {
                        DataPackage* pck_in_fifo = weight_fifo->pop(); //this operation is done i times
                        delete pck_in_fifo; //Deleting pck 
                    }
                    this->mswitchStats.n_weight_fifo_flush++;
                }
                this->mswitchStats.n_weights_receive++; //Tracking the information
                weight_fifo->push(pck); //Inserting new element
                //std::cout << "Weight received by MS " << this->num << ":" << pck->get_data() << std::endl;
            }
     
            else if (pck->get_data_type()==PSUM) {
#ifdef DEBUG_MSWITCH_FUNC
                std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " has received a PSUM from the distribution network" << std::endl;
#endif
                activation_fifo->push(pck); //The psum is push into the activation_fifo
                this->mswitchStats.n_psums_receive++; //Tracking the information
            }
        }
     }
    return;
}

//Perform multiplication with the weight and the activation
DataPackage* MSwitch::perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right) {
    //Extracting the values
    data_t result; // Result of the operation
    result = pck_left->get_data() *  pck_right->get_data(); 
#ifdef DEBUG_MSWITCH_FUNC
    std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ", MSwitch " << this->num << " has performed a multiplication" << std::endl;
#endif
    //std::cout << "[MSWITCH_FUNC] Cycle " << this->local_cycle << ",computing  MSwitch " << this->num << " has performed a multiplication " << pck_left->get_data() << "   " << pck_right->get_data()   << std::endl;

    
    //Creating the result package with the output
    DataPackage* result_pck = new DataPackage (sizeof(data_t), result, PSUM, this->num, this->VN, MULTIPLIER);  //TODO the size of the package corresponds with the data size
    //Adding to the creation list to be deleted afterward
    //this->psums_created.push_back(result_pck);
    this->mswitchStats.n_multiplications++; // Track a multiplication
    return result_pck;


}

//Si es la primera iteracion no recibas del fw link
//Si es la ultima iteracion no envies al fw link
void MSwitch::cycle() { //Computing a cycle
//    std::cout << "MS: " << this->num << ". weight fifo size: " << weight_fifo->size() << " N_Folding: " << n_folding << std::endl;
    local_cycle+=1;
    this->mswitchStats.total_cycles++;
    //   this->receive(inputConnection);
    //if(this->inputForwardingEnabled) {
    //      this->receive(inputForwardingConnection); //Trying to receive from neighbour an input data
   // }


    //Forward psum without managing control
    if(this->direct_forward_psum) {
        this->receive(inputConnection); //Trying to get the psum from the PB
        if(!activation_fifo->isEmpty()) {
            DataPackage* psum = activation_fifo->pop();
            //Creating the psum to forward using the psum received. We have to created a new package since the ART does not need a destination vector.
            DataPackage* psum_fwd = new DataPackage(psum->get_size_package(), psum->get_data(), PSUM, psum->get_source(), this->VN, psum->get_operation_mode());
            delete psum; //Deleting the package received after copying it
            psum_fifo->push(psum_fwd); //Introduce in the fifo to be sent
            this->mswitchStats.n_psum_forwarding_send++;
            this->send(); //Sending to the adder networkQ
	}

    }
    else if(forward_psum && (current_n_folding > 0)) { //The multiplication function is disabled. This MS has to get psums and send them to the parent. 
        this->receive(inputConnection); //Trying to get the psum from the PB
        if(!activation_fifo->isEmpty()) {
            DataPackage* psum = activation_fifo->pop();
            //Creating the psum to forward using the psum received. We have to created a new package since the ART does not need a destination vector.
            DataPackage* psum_fwd = new DataPackage(psum->get_size_package(), psum->get_data(), PSUM, psum->get_source(), this->VN, psum->get_operation_mode());
            delete psum; //Deleting the package received after copying it
            psum_fifo->push(psum_fwd); //Introduce in the fifo to be sent
            this->mswitchStats.n_psum_forwarding_send++;
            this->send(); //Sending to the adder network
            
            current_n_folding+=1; 
            if(current_n_folding == n_folding) {
                current_n_folding = 0;
            }
        }
        
    }

       else if(forward_psum && (current_n_folding==0)){
            DataPackage* zero_psum = new DataPackage(sizeof(data_t), 0, PSUM, 0, this->VN, ADDER); //If it is the first iteration of the window we send a 0.
            psum_fifo->push(zero_psum);
            this->send();
            current_n_folding+=1;
            if(current_n_folding == n_folding) {
                current_n_folding = 0;
            }

        }


    else { //It is a normal MS with folding implementation
        //If the fw link is enabled in this ms and it is not the first window of the row (in whose case all the inputs are fetched from mem)
        Fifo* fifo_to_read; //aux fifo which points to either input_forwarding_fifo or activation_fifo (from mem)
        this->receive(inputConnection);
        if(this->inputForwardingEnabled) {
            this->receive(inputForwardingConnection); //Trying to receive from neighbour an input data
        }
        fifo_to_read = activation_fifo;
        if(this->inputForwardingEnabled && (current_n_windows > 0)) {
            //If inputforwarding is enabled and this is an iteration in which we have to use the forwarding inputs then we take them as inputs
            //assert(!forwarding_input_fifo->isEmpty());
            fifo_to_read = forwarding_input_fifo;
        }
        //Prefetching could be implemented jut changing the condition activation->fifo->isEmpty for a condition that check that the fifo is greater than a certain value to prefetch. 
        if((!fifo_to_read->isEmpty()) && (weight_fifo->size() >= n_folding)) { //If both queues are not empty
            DataPackage* activation = fifo_to_read->pop(); //Get the activation and remove from fifo
            
            DataPackage* weight = weight_fifo->pop(); //get the weight and then pushing again at the end of the fifo
            weight_fifo->push(weight);
            data_t data_read = activation->get_data();
            DataPackage* pck_result = perform_operation_2_operands(activation, weight); //Creating the psum package
            psum_fifo->push(pck_result); //Sending to the output fifo to be read in next cycle
                       //data_t data_read = activation->get_data();
                        //std::cout << "Data Received by MS " << this->num << ": " << data_read << std::endl; 

            this->send(); //Sending to the Adder Network //TODO check flow control
            //if(this->outputForwardingEnabled) { //If the multiplexer is configured to send the current activation to the fw connection
            //    this->forward(activation);
            //}  
            

            //Store the data to forward if this is not the last window and the output is enabled for this MS
      //TODO el problema esta aqui con current_n_windows 
            if((current_n_windows < (n_windows-1)) && this->outputForwardingEnabled) { //Last window is n_windows-1
                DataPackage* activation_forwarded = new DataPackage(activation); //copy to be sent later to the fw link
                forwarding_output_fifo->push(activation_forwarded); //introducing the activation into the fifo to send the data later to the fw link. TODO this shit makes no sense at all.
                activation_forwarded = forwarding_output_fifo->pop();
                this->forward(activation_forwarded); //Sending to the fw link
            }
          
            current_n_folding+=1;
            if(current_n_folding == n_folding) {
                current_n_windows+=1;
                current_n_folding=0;
                if(current_n_windows == n_windows) {
                    current_n_windows = 0;
                }
                //We do not check if we have to update current_n_windows here since the value is neccesary to know the control logic
            }
 
            //Send to the fw link if the first window has been completed (i.e., after the last folding). The last window also send fw links because the folded values has to be sent from the previous clk.
            //However, if it is the last window but it is the last folding of such window current_n_windows is going to be 0 and then it does not send data. 



            
            delete activation; //It is not used anymore since the data was created again. We do this to identify each package as unique for future

        } //End check there are weights and inputs
       
    } //End else forwarding_psum
    
}

void MSwitch::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    out << ind(indent+IND_SIZE) << "\"VN\" : " << this->VN  << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent respo    
}

void MSwitch::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    this->mswitchStats.print(out, indent+IND_SIZE);
    //Printing Fifos 

    out << ind(indent+IND_SIZE) << ",\"ActivationFifo\" : {" << std::endl;
        this->activation_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);         
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"WeightFifo\" : {" << std::endl;
        this->weight_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"PsumFifo\" : {" << std::endl;
        this->psum_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl;; //Take care. Do not print endl here. This is parent responsability
   
    out << ind(indent+IND_SIZE) << "\"ForwardingInputFifo\" : {" << std::endl;
        this->forwarding_input_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl;; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"ForwardingOutputFifo\" : {" << std::endl;
        this->forwarding_output_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}" << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent) << "}"; //TODO put ID




   


}

void MSwitch::printEnergy(std::ofstream& out, unsigned int indent) {
    /*
         This component prints:
             - MSwitch multiplications
             - MSwitch forwardings
             - Counters for the next FIFOs:
                 * activation_fifo: the fifo that is used to receive the activations
                 * forwarding_input_fifo: Fifo used to receive the input activation from neighbour
                 * forwarding_output_fifo: FIFO used to send the input activation to the neighbour
                 * weight_fifo: FIFO used to store the weights
                 * psum_fifo: FIFO used to store the result
   */

    //Multiplier counters
    out << ind(indent) << "MULTIPLIER MULTIPLICATION=" << this->mswitchStats.n_multiplications;
    out << ind(indent) << " FORWARD_PSUM=" << this->mswitchStats.n_psums_receive;
    out << ind(indent) << " CONFIGURATION=" << this->mswitchStats.n_configurations << std::endl;

    //Fifo counters
    activation_fifo->printEnergy(out, indent);
    forwarding_input_fifo->printEnergy(out, indent);
    forwarding_output_fifo->printEnergy(out, indent);
    weight_fifo->printEnergy(out, indent);
    psum_fifo->printEnergy(out, indent);
    
}


