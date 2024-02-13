//Created 19/06/2019 by Francisco Munoz-Martinez

#include "FEASwitch.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include "../../../comm/DataPackage.hpp"
#include "../../../common/utility.hpp"

using namespace std;

/* This class represents the FEASwitch of the MAERI architecture. Basically, the class contains to connections, which   */

FEASwitch::FEASwitch(stonne_id_t id, std::string name, std::size_t level, std::size_t num_in_level, Config stonne_cfg) : Unit(id, name) {
  this->m_level = level;
  this->m_numInLevel = num_in_level;
  this->m_inputPorts = stonne_cfg.m_ASwitchCfg.input_ports;
  this->m_outputPorts = stonne_cfg.m_ASwitchCfg.output_ports;
  //Collecting parameters from the configuration file
  this->m_forwardingPorts = stonne_cfg.m_ASwitchCfg.forwarding_ports;
  this->m_numMs = stonne_cfg.m_MSNetworkCfg.ms_size;
  this->m_buffersCapacity = stonne_cfg.m_ASwitchCfg.buffers_capacity;
  this->m_portWidth = stonne_cfg.m_ASwitchCfg.port_width;
  this->m_latency = stonne_cfg.m_ASwitchCfg.latency;
  //End collecting parameters from the configuration file
  this->m_currentCapacity = 0;
  this->p_inputLeftConnection = NULL;
  this->p_inputRightConnection = NULL;
  this->p_outputConnection = NULL;
  this->p_forwardingConnection = NULL;
  this->m_flDirection = NOT_CONFIGURED;  //This is configured in the first step of the execution
  this->m_configMode = ADD_2_1;          //This is configured in the first step of the execution
  this->m_operationMode = ADDER;         // This is the operation to perform by the AS. By default is an adder.
  this->m_leftChildEnabled = false;
  this->m_rightChildEnabled = false;
  this->m_fwEnabled = false;
  this->p_inputPsumLeftFifo = new Fifo(this->m_buffersCapacity);
  this->p_inputPsumRightFifo = new Fifo(this->m_buffersCapacity);
  this->p_inputFwFifo = new Fifo(this->m_buffersCapacity);
  this->p_inputFwBelowNodesFifo = new Fifo(m_buffersCapacity);
  //  std::cout << "Direccion de memoria antes fw fifo: " << this->input_fw_fifo << std::endl;
  this->p_outputPsumFifo = new Fifo(this->m_buffersCapacity);
  this->p_outputFwFifo = new Fifo(this->m_buffersCapacity);
  this->local_cycle = 0;
  this->m_forwardToMemory = false;
  this->m_currentPsum = 0;
  this->m_forwardToFoldNode = false;

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

FEASwitch::FEASwitch(stonne_id_t id, std::string name, std::size_t level, std::size_t num_in_level, Config stonne_cfg, Connection* inputLeftConnection,
                     Connection* inputRightConnection, Connection* forwardingConnection, Connection* outputConnection, Connection* memoryConnection)
    : FEASwitch(id, name, level, num_in_level, stonne_cfg) {  //Constructor

  this->setInputLeftConnection(inputLeftConnection);
  this->setInputRightConnection(inputRightConnection);
  this->setForwardingConnection(forwardingConnection);
  this->setOutputConnection(outputConnection);
}

FEASwitch::~FEASwitch() {
  delete this->p_inputPsumLeftFifo;
  delete this->p_inputPsumRightFifo;
  delete this->p_inputFwFifo;
  delete this->p_inputFwBelowNodesFifo;
  delete this->p_outputPsumFifo;
  delete this->p_outputFwFifo;
  //for(int i=0; i<psums_created.size(); i++) {
  //delete psums_created[i]; //deleting the psums that have been created by this AS.
  // }
}

//Connection setters

void FEASwitch::setInputLeftConnection(Connection* inputLeftConnection) {
  this->p_inputLeftConnection = inputLeftConnection;
}

void FEASwitch::setInputRightConnection(Connection* inputRightConnection) {
  this->p_inputRightConnection = inputRightConnection;
}

void FEASwitch::setForwardingConnection(Connection* forwardingConnection) {
  this->p_forwardingConnection = forwardingConnection;
}

void FEASwitch::setOutputConnection(Connection* outputConnection) {
  this->p_outputConnection = outputConnection;
}

void FEASwitch::setMemoryConnection(Connection* memoryConnection, std::size_t busID, std::size_t inputID) {
  this->m_busId = busID;
  this->m_inputId = inputID;
  this->p_memoryConnection = memoryConnection;
}

void FEASwitch::setUpNodeForwardingConnection(Connection* upNodeForwardingConnection) {  // Set the connection to the up node that manages the folding
  this->p_upNodeForwardingConnection = upNodeForwardingConnection;
}

void FEASwitch::addInputConnectionToForward(Connection* inputConnectionToForward) {
  this->m_inputConnectionsToForward.push_back(inputConnectionToForward);
}

//Configuration settings (control signals)

//Forwarding link direction for this Adder (SEND or RECEIVE)
void FEASwitch::setForwardingLinkDirection(fl_t fl_direction) {
  assert(this->p_forwardingConnection != NULL);  //Must be a node with a forwardingConnection created
  this->m_fwEnabled = true;                      //Set true this fw link
  this->m_flDirection = fl_direction;
#ifdef DEBUG_ASWITCH_CONFIG
  std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " Fw link enabled: " << fw_enabled << " with direction "
            << get_string_fwlink_direction(fl_direction) << std::endl;
#endif
}

// Configuration mode of the adder (options: ADD_2_1, ADD_3_1, ADD_1_1_PLUS_FW_1_1, FW_2_ or ADD_OR_FORWARD)
void FEASwitch::setConfigurationMode(adderconfig_t config_mode) {
  this->m_configMode = config_mode;
#ifdef DEBUG_ASWITCH_CONFIG
  std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " Conf: " << get_string_adder_configuration(this->config_mode)
            << std::endl;
#endif
}

void FEASwitch::setChildsEnabled(bool left_child_enabled, bool right_child_enabled) {
  this->m_leftChildEnabled = left_child_enabled;
  this->m_rightChildEnabled = right_child_enabled;
#ifdef DEBUG_ASWITCH_CONFIG
  std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " left child enabled: " << this->left_child_enabled << std::endl;
  std::cout << "[ASWITCH_CONFIG] AS " << this->level << ":" << this->num_in_level << " right child enabled: " << this->right_child_enabled << std::endl;
#endif
}

//Operation mode of the adder (options: ADDER, COMPARATOR)
void FEASwitch::setOperationMode(adderoperation_t operation_mode) {
  this->m_operationMode = operation_mode;
}

void FEASwitch::setForwardingToMemoryEnabled(bool forwarding_to_memory) {
  this->m_forwardToMemory = forwarding_to_memory;
  //std::cout << "Switch " << level << ":" << this->num_in_level << " is going to forward data" << std::endl;
}

void FEASwitch::setForwardingToFoldNodeEnabled(bool forwarding_to_fold_node) {
  this->m_forwardToFoldNode = forwarding_to_fold_node;
}

//TODO  //Control here the bw and if there is data. Send the output_psum_fifo and output_fw_fifo to the connections if there is data
void FEASwitch::send() {
  //Sending output_fw_fifo to the fw link. fw_link
  //if((level == 5) && (num_in_level==22)) {
  //   std::cout  << "OUTPUT SIZE EACH CYCLE: " << output_fw_fifo->size() << std::endl;
  // }

  if (!p_outputFwFifo->isEmpty()) {
    assert(this->m_fwEnabled && (this->m_flDirection == SEND));
    std::vector<DataPackage*> vector_to_send_fw_link;
    while (!p_outputFwFifo->isEmpty()) {  //TODO control bw
      DataPackage* pck = p_outputFwFifo->pop();
#ifdef DEBUG_ASWITCH_FUNC
      std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level
                << " has sent a psum to the forwarding link" << std::endl;
#endif

      vector_to_send_fw_link.push_back(pck);
    }
    this->aswitchStats.n_augmented_link_send++;  //Track the information
    this->p_forwardingConnection->send(vector_to_send_fw_link);
  }
  if (!p_outputPsumFifo->isEmpty()) {
    //     std::cout << "DEBUG GENERAL " << this->level << ":" << this->num_in_level << " at cycle " << this->local_cycle << std::endl;
    std::vector<DataPackage*> vector_to_send_parent;
    while (!p_outputPsumFifo->isEmpty()) {
      DataPackage* pck = p_outputPsumFifo->pop();
      vector_to_send_parent.push_back(pck);
    }

    //Sending if there is something
    if (this->m_forwardToMemory) {  //Optimization to send the psum to the memory once it is completed without traverse other adders con fw configuration
      assert(this->m_forwardToFoldNode == false);
#ifdef DEBUG_ASWITCH_FUNC
      std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level
                << " has sent a psum to memory (FORWARDING DATA)" << std::endl;
#endif
      this->aswitchStats.n_memory_send++;  //Track the information
      this->p_memoryConnection->send(vector_to_send_parent);
    }

    else if (this->m_forwardToFoldNode) {
      this->p_upNodeForwardingConnection->send(vector_to_send_parent);
    } else {
#ifdef DEBUG_ASWITCH_FUNC
      std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level << " has sent a psum to the parent"
                << std::endl;
#endif

      this->aswitchStats.n_parent_send++;                     //Track the information
      this->p_outputConnection->send(vector_to_send_parent);  //Send the data to the corresponding output
    }
  }
}

//TODO Controlar el bw
void FEASwitch::receive_childs() {
  if (this->p_inputLeftConnection->existPendingData()) {                                    //If there is data to receive on the left
    std::vector<DataPackage*> data_received_left = this->p_inputLeftConnection->receive();  //Copying the data to receive
    for (int i = 0; i < data_received_left.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
      std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level
                << " has received a psum from input port 0" << std::endl;
#endif
      p_inputPsumLeftFifo->push(data_received_left[i]);  //Inserting to the local queuqe from connection
    }
  }
  if (this->p_inputRightConnection->existPendingData()) {
    std::vector<DataPackage*> data_received_right = this->p_inputRightConnection->receive();
    for (int i = 0; i < data_received_right.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
      std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level
                << " has received a psum from input port 1" << std::endl;
#endif

      p_inputPsumRightFifo->push(data_received_right[i]);
    }
  }
  /*
    if((level == 8) && (num_in_level == 181)) {  // 5 y 22 es el 2:1 que va al vecino 21
        std::cout << "input_psum_right size in receive childs is " << input_psum_right_fifo->size() << std::endl;
        std::cout << "input_psum_left_size in receive childs is " << input_psum_left_fifo->size() << std::endl;
        std::cout << "Left son ENABLEED: " << this->left_child_enabled << std::endl;
        std::cout << "Right son ENABLED: " << this->right_child_enabled << std::endl;
        std::cout << "FEASwitch type is: " << get_string_adder_configuration(config_mode) << std::endl;
    }
        if((level == 6) && (num_in_level == 44)) {
        std::cout << "input_psum_right size 44 SWITCHHHHH " << input_psum_right_fifo->size() << std::endl;
        std::cout << "input_psum_left size 44 SWITCHHHHH " << input_psum_left_fifo->size() << std::endl;
        //std::cout << "FEASwitch type is: " << get_string_adder_configuration(config_mode) << std::endl;
        }
*/
  return;
}

//TODO controlar el bw
void FEASwitch::receive_fwlink() {
  if (this->m_fwEnabled && (m_flDirection == RECEIVE)) {  //If the MS has a forwarding link and it is configured to receive information
    if (this->p_forwardingConnection->existPendingData()) {
      std::vector<DataPackage*> data_received_fw = this->p_forwardingConnection->receive();
      assert(data_received_fw.size() == 1);
      /* if((level == 5) && (num_in_level==21)) {
                if(data_received_fw.size() >= 1) {
                std::cout <<  "Fw link received data" << std::endl;
                }
            }*/

      for (int i = 0; i < data_received_fw.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
        std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level
                  << " has received a psum from the forwarding link" << std::endl;
#endif

        p_inputFwFifo->push(data_received_fw[i]);
      }
    }
    /*if((level == 5) && (num_in_level == 21)) {
        	std::cout << "fw_link_input_fifo size is " << input_fw_fifo->size() << std::endl;
        } */
  }
}

void FEASwitch::receive_below_nodes_fwlinks() {
  std::size_t n_connections_with_data = 0;
  for (int i = 0; i < m_inputConnectionsToForward.size(); i++) {
    if (this->m_inputConnectionsToForward[i]->existPendingData()) {
      n_connections_with_data++;
      std::vector<DataPackage*> data_received_fw = this->m_inputConnectionsToForward[i]->receive();
      assert(data_received_fw.size() == 1);
      for (int i = 0; i < data_received_fw.size(); i++) {
#ifdef DEBUG_ASWITCH_FUNC
        std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level
                  << " has received a psum from the forwarding link of the nodes below" << std::endl;
#endif

        p_inputFwBelowNodesFifo->push(data_received_fw[i]);
      }
    }
  }
  assert(n_connections_with_data <= 1);
}

//Perform operation based on the parameter this->operation_mode
DataPackage* FEASwitch::perform_operation_2_operands(DataPackage* pck_left, DataPackage* pck_right) {
  //Extracting the values
  assert(pck_left->get_vn() == pck_right->get_vn());  // vn must fit

  data_t result;  // Result of the operation
  switch (this->m_operationMode) {
    case ADDER:  //SUM
      result = pck_left->get_data() + pck_right->get_data();
      this->aswitchStats.n_2_1_sums++;  //Track the information
#ifdef DEBUG_ASWITCH_FUNC
      std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level << " has performed a 2:1 sum"
                << std::endl;
#endif

      break;
    case COMPARATOR:                                                                                           //MAX POOL IMPLEMENTATION
      result = (pck_left->get_data() > pck_right->get_data()) ? pck_left->get_data() : pck_right->get_data();  //MAX POOL
      this->aswitchStats.n_2_1_comps++;
      break;

    default:
      assert(false);  // This case must not occur in this type of configuration adder
  }
  //Creating the result package with the output
  DataPackage* result_pck = new DataPackage(sizeof(data_t), result, PSUM, this->m_level, pck_left->get_vn(), this->m_operationMode, pck_left->getRow(),
                                            pck_left->getRow());  //TODO the size of the package corresponds with the data size
  //Adding to the creation list to be deleted afterward
  //    this->psums_created.push_back(result_pck);
  return result_pck;
}

/*
This function performs a 3:1 sum. The comparation 3:1 is not implemented since it does not make much sense in MAERI. If the AS is configured with a 3:1 configuration, it must execute a sum.
This function performs the 3-input sum and creates a package with it which is added to the list this->psums_created by this AS in order to be deleted afterwards. The package is returned 
and the function that called it send the package through the output link.
*/
DataPackage* FEASwitch::perform_operation_3_operands(DataPackage* pck_left, DataPackage* pck_right, DataPackage* pck_forward) {
  assert(pck_left->get_vn() == pck_right->get_vn());
  assert(pck_right->get_vn() == pck_forward->get_vn());  //3 vn ids must fit
  //Calculating the result of the operation
  data_t result;
  switch (this->m_operationMode) {
    case ADDER:  //SUM
#ifdef DEBUG_ASWITCH_FUNC
      std::cout << "[ASWITCH_FUNC] Cycle " << local_cycle << ", FEASwitch " << this->level << ":" << this->num_in_level << " has performed a 3:1 sum"
                << std::endl;
#endif

      result = pck_left->get_data() + pck_right->get_data() + pck_forward->get_data();
      this->aswitchStats.n_3_1_sums++;  //Track the information
      break;
      //case COMPARATOR: //MAX POOL IMPLEMENTATION. Really this does not make much sense here. But it is implemented for possible future uses
      //   result = (pck_left->get_data() > pck_right->get_data()) ? pck_left->get_data() : pck_right->get_data(); //MAX POOL
      //  result = (result > pck_forward->get_data()) ? result : pck_forward->get_data();
      // break;

    default:          //If it's 3:1 must be an adder in this architecture
      assert(false);  // This case must not occur in this type of configuration adder
  }

  //Wrapping over a package
  DataPackage* result_pck = new DataPackage(sizeof(data_t), result, PSUM, this->m_level, pck_left->get_vn(), this->m_operationMode, pck_left->getRow(),
                                            pck_left->getCol());  //TODO the size of the package corresponds with the data size
  //std::cout << "FEASwitch " << level << ":" << num_in_level << " Has computed a value" << std::endl;
  //Adding to the creation list to be deleted afterward
  //this->psums_created.push_back(result_pck);
  return result_pck;
}

//The output of this configuration could be either the output link or the forwarding. It depends on wether the fw link is enabled or not.
//TODO MODELAR LATENCIA !!
void FEASwitch::route_2_1_config() {
  if ((!p_inputPsumLeftFifo->isEmpty()) && (p_inputPsumRightFifo->isEmpty())) {  //If there is no element on the left just forward right
    if (!this->m_rightChildEnabled) {                                            //If the right child is disabled then forward the left
      DataPackage* pck_received = p_inputPsumLeftFifo->pop();                    //Get the data
      DataPackage* pck_to_send = new DataPackage(pck_received);                  //Duplicate the data to delete the memory
      delete pck_received;
      if (this->m_fwEnabled) {  //IF tjhe input is send though the fw link..  //TODO these loops can be better implemented
        p_outputFwFifo->push(pck_to_send);
      } else {
        p_outputPsumFifo->push(pck_to_send);
      }
    }
  }

  if ((!p_inputPsumRightFifo->isEmpty()) && (p_inputPsumLeftFifo->isEmpty())) {  //Forward left
    if (!this->m_leftChildEnabled) {                                             //if left is not enabled then forward right
      DataPackage* pck_received = p_inputPsumRightFifo->pop();
      DataPackage* pck_to_send = new DataPackage(pck_received);
      delete pck_received;

      if (this->m_fwEnabled) {
        p_outputFwFifo->push(pck_to_send);
      }

      else {
        p_outputPsumFifo->push(pck_to_send);
      }
    }
  }

  // Sum the values and send
  if ((!p_inputPsumLeftFifo->isEmpty()) &&
      (!p_inputPsumRightFifo->isEmpty())) {  //If this happens, it is because it is neccesary to sum. The number of operands in both branches must be identical
    assert(this->m_leftChildEnabled);
    assert(this->m_rightChildEnabled);
    //assert(psum_right.size() == psum_left.size());
    DataPackage* pck_left = p_inputPsumLeftFifo->pop();
    DataPackage* pck_right = p_inputPsumRightFifo->pop();
    // Perform the operation
    DataPackage* pck_result = perform_operation_2_operands(pck_left, pck_right);  //pck_result added to the psums_created list in order to be removed afterwards
    if (this->m_fwEnabled) {
      p_outputFwFifo->push(pck_result);  //Sending the result to be read in next cycle //TODO send to forwarding link if required
    } else {
      p_outputPsumFifo->push(pck_result);
    }
    delete pck_left;   //delete the space in memory
    delete pck_right;  //delete the space in memory
  }
}

//TODO MODELAR LATENCIA !!!
void FEASwitch::route_3_1_config() {
  assert(this->m_leftChildEnabled);
  assert(this->m_rightChildEnabled);
  //First we check there is data in all receiving directions (inputleft, inputRight, forwarding link)
  assert(m_fwEnabled && (m_flDirection == RECEIVE));  //The fw link of the AS must be configured correctly

  if (!p_inputPsumLeftFifo->isEmpty() && !p_inputPsumRightFifo->isEmpty() &&
      !p_inputFwFifo->isEmpty()) {  //If there is data (i.e., they all are greater than 0)
    //Compute
    DataPackage* pck_left = p_inputPsumLeftFifo->pop();
    DataPackage* pck_right = p_inputPsumRightFifo->pop();
    DataPackage* pck_forward = p_inputFwFifo->pop();
    //Perform the operation with the 3 operands in each package
    DataPackage* pck_result = perform_operation_3_operands(pck_left, pck_right, pck_forward);
    //std::cout << "Operacion suma " << pck_result->get_data() << std::endl;
    p_outputPsumFifo->push(pck_result);
    //Delete space of memory
    delete pck_left;
    delete pck_right;
    delete pck_forward;
  }
}

//One of the inputs could be empty
void FEASwitch::route_1_1_plus_fw_1_1_config() {
  assert(this->m_leftChildEnabled);
  assert(this->m_rightChildEnabled);
  //    assert(fw_enabled && (fl_direction == SEND)); //The fw link of the AS must be configured correctly
  //Si num_in_level es  par el forwarding link esta a la izquierda (if num_in_level is even, then the fw link is on the left)
  if ((this->m_numInLevel % 2) == 0) {      //If it's even (LEFT)
    if (!p_inputPsumLeftFifo->isEmpty()) {  //If there is left data
      DataPackage* pck_left = p_inputPsumLeftFifo->pop();
      DataPackage* pck_to_send = new DataPackage(pck_left);
      this->p_outputFwFifo->push(pck_to_send);  //Package from left goes to the fw link
      delete pck_left;
    }
    if (!p_inputPsumRightFifo->isEmpty()) {
      DataPackage* pck_right = p_inputPsumRightFifo->pop();
      DataPackage* pck_to_send = new DataPackage(pck_right);
      if (pck_right->get_vn() == 0) {
        assert(false);
      }
      this->p_outputPsumFifo->push(pck_to_send);
      delete pck_right;
    }

  } else {                                  //If it's odd FW link receives the right input and parent the left input
    if (!p_inputPsumLeftFifo->isEmpty()) {  //If there is left data
      DataPackage* pck_left = p_inputPsumLeftFifo->pop();
      DataPackage* pck_to_send = new DataPackage(pck_left);
      this->p_outputPsumFifo->push(pck_to_send);  // The left input goes to the parent
      delete pck_left;
    }
    if (!p_inputPsumRightFifo->isEmpty()) {
      DataPackage* pck_right = p_inputPsumRightFifo->pop();
      DataPackage* pck_to_send = new DataPackage(pck_right);
      this->p_outputFwFifo->push(pck_to_send);  //The right input goes to the fw link
      delete pck_right;
    }
  }

  //Si num_in_evel es impar el forwarding link esta a la derecha (if num_in_level is odd, then the odd link is on the right)
}

//Receive everything from left, everything from right, and fw it to output (parent) (2:2 means 2 inputs 2 outputs, but could consist of more than one value)
void FEASwitch::route_fw_2_2_config() {
  //Forwarding to the outputs which are  used in next cycle. Note we use different loops to insert left and right since could there be different number of psums in each child.
  // If there is nothing psum_left and psum_right will be empty and nothing happens
  //Inserting left inputs
  while (!p_inputPsumLeftFifo->isEmpty()) {
    DataPackage* pck_received = p_inputPsumLeftFifo->pop();
    DataPackage* pck_to_send = new DataPackage(pck_received);
    p_outputPsumFifo->push(pck_to_send);
    delete pck_received;
  }

  //Inserting right inputs
  while (!p_inputPsumRightFifo->isEmpty()) {
    DataPackage* pck_received = p_inputPsumRightFifo->pop();
    DataPackage* pck_to_send = new DataPackage(pck_received);
    p_outputPsumFifo->push(pck_to_send);
    delete pck_received;
  }
}

void FEASwitch::route_fold_config() {
  assert(this->m_forwardToMemory);
  //Receiving psum package from fw links
  DataPackage* pck_received;
  if (!p_inputFwBelowNodesFifo->isEmpty()) {
    pck_received = p_inputFwBelowNodesFifo->pop();
    DataPackage* result;
    if (m_currentPsum == 0) {  //There is no package yet to sum in this iteration
      //Creating package 0
      this->p_temporalRegister = pck_received;

    } else {
      result = perform_operation_2_operands(this->p_temporalRegister, pck_received);
      delete this->p_temporalRegister;
      delete pck_received;
      this->p_temporalRegister = result;
    }

    if (this->m_currentPsum == (this->m_nPsums - 1)) {
      this->p_outputPsumFifo->push(this->p_temporalRegister);
      this->m_currentPsum = 0;

    } else {
      this->m_currentPsum++;
    }
  }
}

//TODO
void FEASwitch::cycle() {
  this->local_cycle += 1;
  this->aswitchStats.total_cycles++;    //Track the information
  this->receive_childs();               //Receive input and right inputs if they exist
  this->receive_fwlink();               // Receive fw input if it exists (if the AS has no fw link enabled, it receives nothing)
  this->receive_below_nodes_fwlinks();  //Only will find data if configuration is FOLD

  switch (this->m_configMode) {  //Routing dependin on the configuration
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

    case FOLD:
      route_fold_config();
      break;
    default:
      assert(false);  //NOT_CONFIGURED INCLUDED
  }

  this->send();  //Send towards the fw link and parent link if there is data pending in output_psum_fifo or output_fw_fifo
}

//Print the configuration of the FEASwitch
void FEASwitch::printConfiguration(std::ofstream& out, std::size_t indent) {
  out << ind(indent) << "{" << std::endl;  //TODO put ID
  out << ind(indent + IND_SIZE) << "\"Configuration\" : \"" << get_string_adder_configuration(this->m_configMode) << "\""
      << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"Augmented_link_enabled\" : " << this->m_fwEnabled << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"Augmented_link_direction\" : \"" << get_string_fwlink_direction(this->m_flDirection) << "\""
      << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"Left_child_enabled\" : " << this->m_leftChildEnabled << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"Right_child_enabled\" : " << this->m_rightChildEnabled << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"BusID\" : " << this->m_busId << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"InputID\" : " << this->m_inputId << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"Send_result_to_fold_node\" : " << this->m_forwardToFoldNode << "," << std::endl;
  out << ind(indent + IND_SIZE) << "\"Send_result_to_memory\" : " << this->m_forwardToMemory << std::endl;
  out << ind(indent) << "}";  //Take care. Do not print endl here. This is parent responsability
}

//Print the statistics obtained during the execution
void FEASwitch::printStats(std::ofstream& out, std::size_t indent) {
  out << ind(indent) << "{" << std::endl;  //TODO put ID
  this->aswitchStats.print(out, indent + IND_SIZE);

  //Printing Fifos
  out << ind(indent + IND_SIZE) << ",\"InputPsumLeftFifo\" : {" << std::endl;
  this->p_inputPsumLeftFifo->printStats(out, indent + IND_SIZE + IND_SIZE);
  out << ind(indent + IND_SIZE) << "}," << std::endl;  //Take care. Do not print endl here. This is parent responsability

  out << ind(indent + IND_SIZE) << "\"InputPsumRightFifo\" : {" << std::endl;
  this->p_inputPsumRightFifo->printStats(out, indent + IND_SIZE + IND_SIZE);
  out << ind(indent + IND_SIZE) << "}," << std::endl;  //Take care. Do not print endl here. This is parent responsability

  out << ind(indent + IND_SIZE) << "\"OutputPsumFifo\" : {" << std::endl;
  this->p_outputPsumFifo->printStats(out, indent + IND_SIZE + IND_SIZE);
  out << ind(indent + IND_SIZE) << "}," << std::endl;
  ;  //Take care. Do not print endl here. This is parent responsability

  out << ind(indent + IND_SIZE) << "\"OutputForwardingFifo\" : {" << std::endl;
  this->p_outputFwFifo->printStats(out, indent + IND_SIZE + IND_SIZE);
  out << ind(indent + IND_SIZE) << "}," << std::endl;
  ;  //Take care. Do not print endl here. This is parent responsability

  out << ind(indent + IND_SIZE) << "\"InputForwardingFifo\" : {" << std::endl;
  this->p_inputFwFifo->printStats(out, indent + IND_SIZE + IND_SIZE);
  out << ind(indent + IND_SIZE) << "}" << std::endl;  //Take care. Do not print endl here. This is parent responsability

  out << ind(indent) << "}";  //Take care. Do not print endl here. This is parent responsability
}

void FEASwitch::printEnergy(std::ofstream& out, std::size_t indent) {
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
  out << ind(indent) << "ADDER ADD_2_1=" << this->aswitchStats.n_2_1_sums;  //SAME LINE
  out << ind(indent) << " ADD_3_1=" << this->aswitchStats.n_3_1_sums;
  out << ind(indent) << " CONFIGURATION=" << this->aswitchStats.n_configurations << std::endl;

  this->p_inputPsumLeftFifo->printEnergy(out, indent);
  this->p_inputPsumRightFifo->printEnergy(out, indent);
  this->p_inputFwFifo->printEnergy(out, indent);
  this->p_outputFwFifo->printEnergy(out, indent);
  this->p_outputPsumFifo->printEnergy(out, indent);
}
