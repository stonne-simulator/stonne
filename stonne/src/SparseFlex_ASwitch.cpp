//Created 19/06/2019 by Francisco Munoz-Martinez

#include "SparseFlex_ASwitch.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include "DataPackage.h"
#include "utility.h"

using namespace std;

/* This class represents the SparseFlex_ASwitch of the MAERI architecture. Basically, the class contains to connections, which   */

SparseFlex_ASwitch::SparseFlex_ASwitch(id_t id, std::string name, unsigned int level, unsigned int num_in_level, Config stonne_cfg) : Unit(id, name) {
    this->level = level;
    this->num_in_level = num_in_level;
    this->input_ports = stonne_cfg.m_ASwitchCfg.input_ports;
    this->output_ports = stonne_cfg.m_ASwitchCfg.output_ports;
    //Collecting parameters from the configuration file
    this->forwarding_ports = stonne_cfg.m_ASwitchCfg.forwarding_ports;
    this->num_ms = stonne_cfg.m_MSNetworkCfg.ms_size;
    this->buffers_capacity = stonne_cfg.m_ASwitchCfg.buffers_capacity;
    this->port_width = stonne_cfg.m_ASwitchCfg.port_width;
    this->latency = stonne_cfg.m_ASwitchCfg.latency;
    //End collecting parameters from the configuration file
    this->current_capacity = 0;
    this->inputLeftConnection = NULL;
    this->inputRightConnection = NULL;
    this->outputConnection = NULL;
    this->forwardingConnection = NULL;
    this->fl_direction = NOT_CONFIGURED;  //This is configured in the first step of the execution
    this->config_mode = ADD_2_1;  //This is configured in the first step of the execution
    this->operation_mode = ADDER;   // This is the operation to perform by the AS. By default is an adder.
    this->left_child_enabled = false;
    this->right_child_enabled = false;
    this->fw_enabled = false; 
    this->input_psum_left_fifo = new Fifo(this->buffers_capacity);
    this->input_psum_right_fifo = new Fifo(this->buffers_capacity);
    this->input_fw_fifo = new Fifo(this->buffers_capacity);
  //  std::cout << "Direccion de memoria antes fw fifo: " << this->input_fw_fifo << std::endl; 
    this->output_psum_fifo = new Fifo(this->buffers_capacity);
    this->output_fw_fifo = new Fifo(this->buffers_capacity);
    this->local_cycle=0;
    this->forward_to_memory=false;

/*
    //Forwarding to memory flags enabled by hand 
    if((level==2) && (num_in_level==0)) {
        this->forward_to_memory=true;
    }

    if((level==3) && (num_in_level==3)) {
        this->forward_to_memory=true;
    }
   // if((level==2) && (num_in_level==2)) {
   //     this->forward_to_memory=true;
   // }
   // if((level==1) && (num_in_level==1)) {
   //     this->forward_to_memory=true;
   // }


   // if((level==2) && (num_in_level==3)) {
   //     this->forward_to_memory=true;
   // }
  
//    if((level==3) && (num_in_level==3)) {
  //      this->forward_to_memory=true;
  //  }
   //  if((level==3) && (num_in_level==5)) {
    //    this->forward_to_memory=true;
   // } 

*/
   
}

SparseFlex_ASwitch::SparseFlex_ASwitch(id_t id, std::string name, unsigned int level, unsigned int num_in_level, Config stonne_cfg, Connection* inputLeftConnection, 
Connection* inputRightConnection, Connection* forwardingConnection, Connection* outputConnection, Connection* memoryConnection) : SparseFlex_ASwitch(id, name, level, num_in_level, stonne_cfg) { //Constructor

    this->setInputLeftConnection(inputLeftConnection);
    this->setInputRightConnection(inputRightConnection);
    this->setForwardingConnection(forwardingConnection);
    this->setOutputConnection(outputConnection);
}

SparseFlex_ASwitch::~SparseFlex_ASwitch() {
    delete this->input_psum_left_fifo;
    delete this->input_psum_right_fifo;
    delete this->input_fw_fifo;
    delete this->output_psum_fifo;
    delete this->output_fw_fifo;
    //for(int i=0; i<psums_created.size(); i++) {
        //delete psums_created[i]; //deleting the psums that have been created by this AS.
   // }
}

void SparseFlex_ASwitch::resetSignals() {
    //End collecting parameters from the configuration file
    this->current_capacity = 0;
    this->fl_direction = NOT_CONFIGURED;  //This is configured in the first step of the execution
    this->config_mode = ADD_2_1;  //This is configured in the first step of the execution
    this->operation_mode = ADDER;   // This is the operation to perform by the AS. By default is an adder.
    this->left_child_enabled = false;
    this->right_child_enabled = false;
    this->fw_enabled = false; 
    this->forward_to_memory=false;
          while(!input_psum_left_fifo->isEmpty()) {
        delete input_psum_left_fifo->pop();
    }

      while(!input_psum_right_fifo->isEmpty()) {
        delete input_psum_right_fifo->pop();
    }

      while(!output_psum_fifo->isEmpty()) {
        delete output_psum_fifo->pop();
    }

      while(!input_fw_fifo->isEmpty()) {
        delete input_fw_fifo->pop();
    }

      while(!output_fw_fifo->isEmpty()) {
        delete output_fw_fifo->pop();
    }

}

//Connection setters

void SparseFlex_ASwitch::setInputLeftConnection(Connection* inputLeftConnection) {
    this->inputLeftConnection = inputLeftConnection;
}

void SparseFlex_ASwitch::setInputRightConnection(Connection* inputRightConnection) {
    this->inputRightConnection = inputRightConnection;
}

void SparseFlex_ASwitch::setForwardingConnection(Connection* forwardingConnection) {
    this->forwardingConnection = forwardingConnection;
}

void SparseFlex_ASwitch::setOutputConnection(Connection* outputConnection) {
    this->outputConnection = outputConnection;
}

void SparseFlex_ASwitch::setMemoryConnection(Connection* memoryConnection,  unsigned int busID, unsigned int inputID) {
    this->busID = busID;
    this->inputID = inputID;
    this->memoryConnection = memoryConnection;
}



//Configuration settings (control signals)

//Forwarding link direction for this Adder (SEND or RECEIVE)
void SparseFlex_ASwitch::setForwardingLinkDirection(fl_t fl_direction) {
    assert(this->forwardingConnection != NULL); //Must be a node with a forwardingConnection created
    this->fw_enabled = true; //Set true this fw link
    this->fl_direction = fl_direction;
#ifdef DEBUG_ASWITCH_CONFIG
    std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " Fw link enabled: " << fw_enabled << " with direction " << get_string_fwlink_direction(fl_direction) << std::endl;
#endif

}

// Configuration mode of the adder (options: ADD_2_1, ADD_3_1, ADD_1_1_PLUS_FW_1_1, FW_2_ or ADD_OR_FORWARD)
void SparseFlex_ASwitch::setConfigurationMode(adderconfig_t config_mode) {
    this->config_mode = config_mode;
    this->aswitchStats.n_configurations++;
#ifdef DEBUG_ASWITCH_CONFIG
    std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " Conf: " << get_string_adder_configuration(this->config_mode)  << std::endl;
#endif
}

void SparseFlex_ASwitch::setChildsEnabled(bool left_child_enabled, bool right_child_enabled) {
    this->left_child_enabled = left_child_enabled;
    this->right_child_enabled = right_child_enabled;
#ifdef DEBUG_ASWITCH_CONFIG
    std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " left child enabled: " << this->left_child_enabled << std::endl;
    std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " right child enabled: " << this->right_child_enabled << std::endl;
#endif
}

//Operation mode of the adder (options: ADDER, COMPARATOR)
void SparseFlex_ASwitch::setOperationMode(adderoperation_t operation_mode) {
    this->operation_mode = operation_mode;
}

void SparseFlex_ASwitch::setForwardingToMemoryEnabled(bool forwarding_to_memory) {
    this->forward_to_memory=forwarding_to_memory;
    //std::cout << "Switch " << level << ":" << this->num_in_level << " is going to forward data" << std::endl;
}

//TODO  //Control here the bw and if there is data. Send the output_psum_fifo and output_fw_fifo to the connections if there is data
void SparseFlex_ASwitch::send() {
    //Sending output_fw_fifo to the fw link. fw_link
         //if((level == 5) && (num_in_level==22)) {
         //   std::cout  << "OUTPUT SIZE EACH CYCLE: " << output_fw_fifo->size() << std::endl;
       // }

    if(!output_fw_fifo->isEmpty()) {
        assert(this->fw_enabled && (this->fl_direction == SEND));
        std::vector<DataPackage*> vector_to_send_fw_link;
        while(!output_fw_fifo->isEmpty()) { //TODO control bw 
            DataPackage* pck = output_fw_fifo->pop();

#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has sent a psum to the forwarding link" << std::endl;
#endif
            
            vector_to_send_fw_link.push_back(pck);
        }
            this->aswitchStats.n_augmented_link_send++; //Track the information
            this->forwardingConnection->send(vector_to_send_fw_link);

    }
    if(!output_psum_fifo->isEmpty()) {
   //     std::cout << "DEBUG GENERAL " << this->level << ":" << this->num_in_level << " at cycle " << this->local_cycle << std::endl;
        std::vector<DataPackage*> vector_to_send_parent;
        while(!output_psum_fifo->isEmpty()) {
             DataPackage* pck = output_psum_fifo->pop();
             vector_to_send_parent.push_back(pck);
        }

        //Sending if there is something
        if(this->forward_to_memory) { //Optimization to send the psum to the memory once it is completed without traverse other adders con fw configuration
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has sent a psum to memory (FORWARDING DATA)" << std::endl;
#endif
            this->aswitchStats.n_memory_send++; //Track the information
            this->memoryConnection->send(vector_to_send_parent); 
        }
        else { 
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has sent a psum to the parent" << std::endl;
#endif

            this->aswitchStats.n_parent_send++; //Track the information
            this->outputConnection->send(vector_to_send_parent); //Send the data to the corresponding output
        }
    }
}

//TODO Controlar el bw
void SparseFlex_ASwitch::receive_childs() { 
    if(this->inputLeftConnection->existPendingData()) { //If there is data to receive on the left
    	std::vector<DataPackage*> data_received_left = this->inputLeftConnection->receive(); //Copying the data to receive
        for(int i=0; i<data_received_left.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has received a psum from input port 0" << std::endl;
#endif
            input_psum_left_fifo->push(data_received_left[i]); //Inserting to the local queuqe from connection
        }
    }
    if(this->inputRightConnection->existPendingData()) {
        std::vector<DataPackage*> data_received_right = this->inputRightConnection->receive();
        for(int i=0; i<data_received_right.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has received a psum from input port 1" << std::endl;
#endif

            input_psum_right_fifo->push(data_received_right[i]);
        }
    }
/*
    if((level == 8) && (num_in_level == 181)) {  // 5 y 22 es el 2:1 que va al vecino 21
        std::cout << "input_psum_right size in receive childs is " << input_psum_right_fifo->size() << std::endl;
        std::cout << "input_psum_left_size in receive childs is " << input_psum_left_fifo->size() << std::endl;
        std::cout << "Left son ENABLEED: " << this->left_child_enabled << std::endl;
        std::cout << "Right son ENABLED: " << this->right_child_enabled << std::endl;
        std::cout << "SparseFlex_ASwitch type is: " << get_string_adder_configuration(config_mode) << std::endl;
    }
        if((level == 6) && (num_in_level == 44)) {
        std::cout << "input_psum_right size 44 SWITCHHHHH " << input_psum_right_fifo->size() << std::endl;
        std::cout << "input_psum_left size 44 SWITCHHHHH " << input_psum_left_fifo->size() << std::endl;
        //std::cout << "SparseFlex_ASwitch type is: " << get_string_adder_configuration(config_mode) << std::endl;
        }
*/
    return;
}

//TODO controlar el bw
void SparseFlex_ASwitch::receive_fwlink() {
    if(this->fw_enabled && (fl_direction == RECEIVE)) { //If the MS has a forwarding link and it is configured to receive information
        if(this->forwardingConnection->existPendingData()) {
            std::vector<DataPackage*> data_received_fw = this->forwardingConnection->receive();
            assert(data_received_fw.size() == 1);
           /* if((level == 5) && (num_in_level==21)) {
                if(data_received_fw.size() >= 1) {
                std::cout <<  "Fw link received data" << std::endl;
                }
            }*/

            for(int i=0; i<data_received_fw.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
                std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has received a psum from the forwarding link" << std::endl;
#endif

                input_fw_fifo->push(data_received_fw[i]);
            }
        }
         /*if((level == 5) && (num_in_level == 21)) {
        	std::cout << "fw_link_input_fifo size is " << input_fw_fifo->size() << std::endl;
        } */

    }
}

//Perform operation based on the parameter this->operation_mode
DataPackage* SparseFlex_ASwitch::perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right) {
    //Extracting the values
    //assert(pck_left->get_vn() == pck_right->get_vn()); // vn must fit
    
    data_t result; // Result of the operation
    switch(this->operation_mode) {
        case ADDER: //SUM
            result = pck_left->get_data() +  pck_right->get_data();
            this->aswitchStats.n_2_1_sums++;  //Track the information
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has performed a 2:1 sum" << std::endl;
#endif

            break;
        case COMPARATOR: //MAX POOL IMPLEMENTATION 
            result = (pck_left->get_data() > pck_right->get_data()) ? pck_left->get_data() : pck_right->get_data(); //MAX POOL
            this->aswitchStats.n_2_1_comps++;
            break;

        default:
            assert(false); // This case must not occur in this type of configuration adder
    }
    //Creating the result package with the output
    DataPackage* result_pck = new DataPackage (sizeof(data_t), result, PSUM, pck_left->get_source(), pck_left->get_vn(), this->operation_mode, pck_right->getRow(), pck_left->getCol());  //TODO the size of the package corresponds with the data size
    //Adding to the creation list to be deleted afterward
//    this->psums_created.push_back(result_pck);
    return result_pck;
     

}

/*
This function performs a 3:1 sum. The comparation 3:1 is not implemented since it does not make much sense in MAERI. If the AS is configured with a 3:1 configuration, it must execute a sum.
This function performs the 3-input sum and creates a package with it which is added to the list this->psums_created by this AS in order to be deleted afterwards. The package is returned 
and the function that called it send the package through the output link.
*/
DataPackage* SparseFlex_ASwitch::perform_operation_3_operands(DataPackage* pck_left, DataPackage* pck_right, DataPackage* pck_forward) {
    if(pck_left->get_vn() != pck_right->get_vn()) {
        std::cout << "Ha petado en el SparseFlex_ASwitch: " << "(" << this->level << ":" << this->num_in_level << ")" << "Left VN=" << pck_left->get_vn() << " Right VN=" << pck_right->get_vn() <<  std::endl;
    }
    assert(pck_left->get_vn() == pck_right->get_vn()); 
    assert(pck_right->get_vn() == pck_forward->get_vn()); //3 vn ids must fit
    //Calculating the result of the operation
    data_t result;
    switch(this->operation_mode) {
        case ADDER: //SUM
#ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", SparseFlex_ASwitch " << this->level << ":" << this->num_in_level << " has performed a 3:1 sum" << std::endl;
#endif

            result = pck_left->get_data() +  pck_right->get_data() +  pck_forward->get_data();
            this->aswitchStats.n_3_1_sums++; //Track the information
            break;
        //case COMPARATOR: //MAX POOL IMPLEMENTATION. Really this does not make much sense here. But it is implemented for possible future uses
         //   result = (pck_left->get_data() > pck_right->get_data()) ? pck_left->get_data() : pck_right->get_data(); //MAX POOL
          //  result = (result > pck_forward->get_data()) ? result : pck_forward->get_data();
           // break;

        default: //If it's 3:1 must be an adder in this architecture
            assert(false); // This case must not occur in this type of configuration adder
    }

    //Wrapping over a package
    DataPackage* result_pck = new DataPackage (sizeof(data_t), result, PSUM, this->level, pck_left->get_vn(), this->operation_mode);  //TODO the size of the package corresponds with the data size
    //std::cout << "SparseFlex_ASwitch " << level << ":" << num_in_level << " Has computed a value" << std::endl;
    //Adding to the creation list to be deleted afterward
    //this->psums_created.push_back(result_pck);
    return result_pck;

    
}

//The output of this configuration could be either the output link or the forwarding. It depends on wether the fw link is enabled or not.
//TODO MODELAR LATENCIA !!
void SparseFlex_ASwitch::route_2_1_config() {
    if((!input_psum_left_fifo->isEmpty()) && (input_psum_right_fifo->isEmpty()))  { //If there is no element on the left just forward right
        if(!this->right_child_enabled) { //If the right child is disabled then forward the left
            DataPackage* pck_received = input_psum_left_fifo->pop(); //Get the data
            DataPackage* pck_to_send = new DataPackage(pck_received); //Duplicate the data to delete the memory 
            delete pck_received;
            if(this->fw_enabled) { //IF tjhe input is send though the fw link..  //TODO these loops can be better implemented
                output_fw_fifo->push(pck_to_send);
            }
            else {
                output_psum_fifo->push(pck_to_send);
            }
        }
    }

    if((!input_psum_right_fifo->isEmpty()) && (input_psum_left_fifo->isEmpty())) { //Forward left
        if(!this->left_child_enabled) { //if left is not enabled then forward right
            DataPackage* pck_received = input_psum_right_fifo->pop();
            DataPackage* pck_to_send = new DataPackage(pck_received);
            delete pck_received;
        
            if(this->fw_enabled) {
                output_fw_fifo->push(pck_to_send);
            }

            else {
                output_psum_fifo->push(pck_to_send);
            }
        }
    }

    // Sum the values and send
    if((!input_psum_left_fifo->isEmpty()) && (!input_psum_right_fifo->isEmpty())) { //If this happens, it is because it is neccesary to sum. The number of operands in both branches must be identical
            assert(this->left_child_enabled);
            assert(this->right_child_enabled);
        //assert(psum_right.size() == psum_left.size());
            DataPackage* pck_left = input_psum_left_fifo->pop();
            DataPackage* pck_right = input_psum_right_fifo->pop();
            // Perform the operation
            DataPackage* pck_result = perform_operation_2_operands(pck_left, pck_right); //pck_result added to the psums_created list in order to be removed afterwards  
            if(this->fw_enabled) {
                output_fw_fifo->push(pck_result);  //Sending the result to be read in next cycle //TODO send to forwarding link if required
            }
            else {
                output_psum_fifo->push(pck_result);
            }
            delete pck_left; //delete the space in memory
            delete pck_right;  //delete the space in memory
        }
       
}


//TODO MODELAR LATENCIA !!!
void SparseFlex_ASwitch::route_3_1_config() {
    assert(this->left_child_enabled);
    assert(this->right_child_enabled);
    //First we check there is data in all receiving directions (inputleft, inputRight, forwarding link)    
    assert(fw_enabled && (fl_direction == RECEIVE)); //The fw link of the AS must be configured correctly

    if(!input_psum_left_fifo->isEmpty() && !input_psum_right_fifo->isEmpty() && !input_fw_fifo->isEmpty()) { //If there is data (i.e., they all are greater than 0)
        //Compute  
        DataPackage* pck_left = input_psum_left_fifo->pop();
        DataPackage* pck_right = input_psum_right_fifo->pop();
        DataPackage* pck_forward = input_fw_fifo->pop();
        //Perform the operation with the 3 operands in each package
        DataPackage* pck_result = perform_operation_3_operands(pck_left, pck_right, pck_forward);
        //std::cout << "Operacion suma " << pck_result->get_data() << std::endl;
        output_psum_fifo->push(pck_result);
        //Delete space of memory 
        delete pck_left;
        delete pck_right;
        delete pck_forward;
    }
}

void SparseFlex_ASwitch::route_sort_tree_config() {
    //Left flows
    if(!input_psum_left_fifo->isEmpty() && input_psum_right_fifo->isEmpty()) {
        DataPackage* pck_left = input_psum_left_fifo->pop();
	output_psum_fifo->push(pck_left);
    }

    //Right flows
    else if(input_psum_left_fifo->isEmpty() && !input_psum_right_fifo->isEmpty() ) {
        DataPackage* pck_right = input_psum_right_fifo->pop();
	output_psum_fifo->push(pck_right);
    }

    //Compare and flow
    else if (!input_psum_left_fifo->isEmpty() && !input_psum_right_fifo->isEmpty()) {
        DataPackage* pck_left = input_psum_left_fifo->front();
	DataPackage* pck_right = input_psum_right_fifo->front();
	//Left flows
	if(pck_left->getCol() < pck_right->getCol()) {
            input_psum_left_fifo->pop();
            output_psum_fifo->push(pck_left);	    
	}

	//Right flows
	else if(pck_left->getCol() > pck_right->getCol()) {
            input_psum_right_fifo->pop();
	    output_psum_fifo->push(pck_right);
	}

	else { //Both indexes are the same
            DataPackage* pck_result = perform_operation_2_operands(pck_left, pck_right);    
	    output_psum_fifo->push(pck_result);
	    input_psum_left_fifo->pop();
	    input_psum_right_fifo->pop();
	    delete pck_left;
	    delete pck_right;
	}

    }
}

//One of the inputs could be empty
void SparseFlex_ASwitch::route_1_1_plus_fw_1_1_config() {
      assert(this->left_child_enabled);
      assert(this->right_child_enabled);
//    assert(fw_enabled && (fl_direction == SEND)); //The fw link of the AS must be configured correctly
    //Si num_in_level es  par el forwarding link esta a la izquierda (if num_in_level is even, then the fw link is on the left)
    if((this->num_in_level % 2) == 0) { //If it's even (LEFT)
        if(!input_psum_left_fifo->isEmpty()) { //If there is left data
            DataPackage* pck_left = input_psum_left_fifo->pop();
            DataPackage* pck_to_send = new DataPackage(pck_left);
            this->output_fw_fifo->push(pck_to_send); //Package from left goes to the fw link 
            delete pck_left;
        }
        if(!input_psum_right_fifo->isEmpty()) {
            DataPackage* pck_right = input_psum_right_fifo->pop(); 
            DataPackage* pck_to_send = new DataPackage (pck_right);
            if(pck_right->get_vn()==0) {
                assert(false);
            }
            this->output_psum_fifo->push(pck_to_send);
            delete pck_right;
        }

    }
    else { //If it's odd FW link receives the right input and parent the left input
        if(!input_psum_left_fifo->isEmpty()) { //If there is left data
            DataPackage* pck_left = input_psum_left_fifo->pop();
            DataPackage* pck_to_send = new DataPackage(pck_left);
            this->output_psum_fifo->push(pck_to_send); // The left input goes to the parent
            delete pck_left;
        }
        if(!input_psum_right_fifo->isEmpty()) {
            DataPackage* pck_right = input_psum_right_fifo->pop();   
            DataPackage* pck_to_send = new DataPackage(pck_right);
            this->output_fw_fifo->push(pck_to_send); //The right input goes to the fw link
            delete pck_right;
        }
    }

    //Si num_in_evel es impar el forwarding link esta a la derecha (if num_in_level is odd, then the odd link is on the right)
 
}

//Receive everything from left, everything from right, and fw it to output (parent) (2:2 means 2 inputs 2 outputs, but could consist of more than one value)
void SparseFlex_ASwitch::route_fw_2_2_config() {
    //Forwarding to the outputs which are  used in next cycle. Note we use different loops to insert left and right since could there be different number of psums in each child. 
    // If there is nothing psum_left and psum_right will be empty and nothing happens
    //Inserting left inputs
    while(!input_psum_left_fifo->isEmpty()) {
        DataPackage* pck_received = input_psum_left_fifo->pop();
        DataPackage* pck_to_send = new DataPackage(pck_received);
        output_psum_fifo->push(pck_to_send);
        delete pck_received;
    }

    //Inserting right inputs
    while(!input_psum_right_fifo->isEmpty()) {
        DataPackage* pck_received = input_psum_right_fifo->pop();
        DataPackage* pck_to_send = new DataPackage(pck_received);
        output_psum_fifo->push(pck_to_send);
        delete pck_received;
    }
}

//TODO
void SparseFlex_ASwitch::cycle() {
    this->local_cycle+=1; 
    this->aswitchStats.total_cycles++; //Track the information
    this->receive_childs(); //Receive input and right inputs if they exist
    this->receive_fwlink(); // Receive fw input if it exists (if the AS has no fw link enabled, it receives nothing)

    switch(this->config_mode) { //Routing dependin on the configuration
        case ADD_2_1:  
            route_2_1_config();
            break;

        case ADD_3_1:
            route_3_1_config();
            break;

        case ADD_1_1_PLUS_FW_1_1:
            route_1_1_plus_fw_1_1_config();
            break;

        case FW_2_2:
            route_fw_2_2_config();
            break;

	case SORT_TREE:
	    route_sort_tree_config();
	    break;

        default:
            assert(false);  //NOT_CONFIGURED INCLUDED 

    } 

    this->send(); //Send towards the fw link and parent link if there is data pending in output_psum_fifo or output_fw_fifo

}

//Print the configuration of the SparseFlex_ASwitch
void SparseFlex_ASwitch::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    out << ind(indent+IND_SIZE) << "\"Configuration\" : \"" <<get_string_adder_configuration(this->config_mode) << "\"" << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Augmented_link_enabled\" : " << this->fw_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Augmented_link_direction\" : \"" << get_string_fwlink_direction(this->fl_direction) << "\"" << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Left_child_enabled\" : " << this->left_child_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Right_child_enabled\" : " << this->right_child_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"BusID\" : " << this->busID << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"InputID\" : " << this->inputID << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Send_result_to_memory\" : " << this->forward_to_memory  << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

//Print the statistics obtained during the execution
void SparseFlex_ASwitch::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    this->aswitchStats.print(out, indent+IND_SIZE);
  
    //Printing Fifos
    out << ind(indent+IND_SIZE) << ",\"InputPsumLeftFifo\" : {" << std::endl;
    this->input_psum_left_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"InputPsumRightFifo\" : {" << std::endl;
        this->input_psum_right_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"OutputPsumFifo\" : {" << std::endl;
        this->output_psum_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl;; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"OutputForwardingFifo\" : {" << std::endl;
        this->output_fw_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl;; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"InputForwardingFifo\" : {" << std::endl;
        this->input_fw_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}" << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void SparseFlex_ASwitch::printEnergy(std::ofstream& out, unsigned int indent){
    /* 
     This component prints:
         - Number of 2_1 sums
         - Number of 3_1 sums
         - Number of reads and writes to the next fifos:
             * input_psum_left_fifo: fifo to receive data from the left child
             * input_psum_right_fifo: fifo to receive data from the right child
             * input_fw_fifo: fifo to receive data from the fw link
             * output_fw_fifo: fifo to send data to the fw link
             * output_psum_fifo: fifo to send data to the parent
    */
    out << ind(indent) << "ADDER ADD_2_1=" <<  this->aswitchStats.n_2_1_sums; //SAME LINE
    out << ind(indent) << " ADD_3_1=" <<  this->aswitchStats.n_3_1_sums;
    out << ind(indent) << " CONFIGURATION=" << this->aswitchStats.n_configurations << std::endl;

    
    this->input_psum_left_fifo->printEnergy(out, indent);
    this->input_psum_right_fifo->printEnergy(out, indent);
    this->input_fw_fifo->printEnergy(out, indent);
    this->output_fw_fifo->printEnergy(out, indent);
    this->output_psum_fifo->printEnergy(out, indent);

}
