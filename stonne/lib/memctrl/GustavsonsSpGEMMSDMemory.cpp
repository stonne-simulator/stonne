
#include "GustavsonsSpGEMMSDMemory.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include "common/utility.hpp"

GustavsonsSpGEMMSDMemory::GustavsonsSpGEMMSDMemory(stonne_id_t id, std::string name, Config stonne_cfg, Connection* write_connection, Memory<float>& mem)
    : MemoryController(id, name), mem(mem) {
  this->write_connection = write_connection;

  //Collecting parameters from the configuration file
  this->num_ms = stonne_cfg.m_MSNetworkCfg.ms_size;  //Used to send data
  this->n_read_ports = stonne_cfg.m_SDMemoryCfg.n_read_ports;
  this->n_write_ports = stonne_cfg.m_SDMemoryCfg.n_write_ports;
  this->write_buffer_capacity = stonne_cfg.m_SDMemoryCfg.write_buffer_capacity;
  this->port_width = stonne_cfg.m_SDMemoryCfg.port_width;

  this->weight_dram_location = stonne_cfg.m_SDMemoryCfg.weight_address;
  this->input_dram_location = stonne_cfg.m_SDMemoryCfg.input_address;
  this->output_dram_location = stonne_cfg.m_SDMemoryCfg.output_address;
  this->data_width = stonne_cfg.m_SDMemoryCfg.data_width;
  this->n_write_mshr = stonne_cfg.m_SDMemoryCfg.n_write_mshr;

  //End collecting parameters from the configuration file
  //Initializing parameters
  this->ms_size_per_input_port = this->num_ms / this->n_read_ports;
  this->write_fifo = new Fifo(write_buffer_capacity);
  for (int i = 0; i < (this->n_read_ports); i++) {
    Fifo* read_fi = new Fifo(this->write_buffer_capacity);
    Fifo* psum_fi = new Fifo(this->write_buffer_capacity);
    input_fifos.push_back(read_fi);
    psum_fifos.push_back(psum_fi);
    this->sdmemoryStats.n_SRAM_read_ports_weights_use.push_back(0);  //To track information
    this->sdmemoryStats.n_SRAM_read_ports_inputs_use.push_back(0);   //To track information
    this->sdmemoryStats.n_SRAM_read_ports_psums_use.push_back(0);    //To track information
  }
  for (int i = 0; i < (this->n_write_ports) * 2; i++) {       //To track information
    this->sdmemoryStats.n_SRAM_write_ports_use.push_back(0);  //To track information
  }                                                           //To track information
  this->configuration_done = false;
  this->stationary_distributed = false;
  this->stationary_finished = false;
  this->stream_finished = false;
  this->execution_finished = false;
  this->metadata_loaded = false;
  this->layer_loaded = false;
  this->local_cycle = 0;
  this->current_state = CONFIGURING;
  this->current_output = 0;
  this->output_size = 0;
  this->sta_iter_completed = false;
  this->current_output_iteration = 0;
  this->output_size_iteration = 0;
  this->n_ones_sta_matrix = 0;
  this->n_ones_str_matrix = 0;
  this->STA_complete = false;
  //Outer product pointers
  this->current_MK = 0;
  this->current_MK_row_pointer = 0;
  this->current_MK_col_id = 0;
  this->current_KN = 0;
  this->current_KN_row_pointer = 0;
  this->current_KN_col_id = 0;
  this->MK_number_nnz = 0;
  this->multipliers_used = 0;
  this->STR_complete = false;
  this->last_sta_iteration_completed = false;
  this->n_str_data_sent = 0;
  this->n_str_data_received = 0;
  this->multiplication_phase_finished = false;
  this->sort_col_id = 0;
  this->sort_row_id = 0;
  this->sort_down_last_iteration_finished = false;
  this->sort_down_iteration_finished = false;
  this->sort_up_iteration_finished = false;
  this->sort_up_received_first_value = false;
  this->sort_sub_block_id = 0;
  this->sort_num_blocks = 0;
  for (int i = 0; i < this->num_ms; i++) {
    vnat_table.push_back(-1);  //Initializing table with rows of sta data
    ms_group.push_back(-1);
    std::queue<DataPackage*> FIFO;
    buffer_sync.push_back(FIFO);
  }
  this->n_values_stored = 0;
  this->swap_memory_enabled = false;
  this->current_sorting_iteration = 0;
  this->n_str_req_recv = 0;
  this->n_str_req_sent = 0;
}

GustavsonsSpGEMMSDMemory::~GustavsonsSpGEMMSDMemory() {
  delete write_fifo;
  //Deleting the input ports
  for (int i = 0; i < (this->n_read_ports); i++) {
    delete input_fifos[i];
    delete psum_fifos[i];
  }
  //delete[] data_package_array_sync;
}

void GustavsonsSpGEMMSDMemory::setWriteConnections(std::vector<Connection*> write_port_connections) {
  this->write_port_connections = write_port_connections;  //Copying all the poiners
                                                          //assert(this->write_port_connections.size()==this->n_write_ports);
}

void GustavsonsSpGEMMSDMemory::setReadConnections(std::vector<Connection*> read_connections) {
  assert(read_connections.size() == (n_read_ports));  //Checking that the number of input ports is valid.
  std::cout << "Number of read connections: " << read_connections.size() << std::endl;
  this->read_connections = read_connections;  //Copying all the pointers
}

void GustavsonsSpGEMMSDMemory::setLayer(DNNLayer* dnn_layer, address_t MK_address, address_t KN_address, address_t output_address, Dataflow dataflow) {
  this->dnn_layer = dnn_layer;
  assert(this->dnn_layer->get_layer_type() == SPARSE_DENSE);  // This controller only supports GEMM with one sparse and one dense
  //this->dataflow = dataflow;

  this->output_address = output_address;
  this->layer_loaded = true;

  //Loading parameters according to the equivalence between CNN layer and GEMM. This is done
  //in this way to keep the same interface.
  this->M = this->dnn_layer->get_N();
  this->K = this->dnn_layer->get_C();  //Be careful. K in GEMMs (SIGMA taxonomy) is not the same as K in CNN taxonomy (number of filters)
  this->N = this->dnn_layer->get_K();  //In this case both parameters match each other.
  std::cout << "Value of M: " << this->M << std::endl;
  std::cout << "Value of K: " << this->K << std::endl;
  std::cout << "Value of N: " << this->N << std::endl;
  sdmemoryStats.dataflow = dataflow;

  this->MK_address = MK_address;
  this->KN_address = KN_address;

  //  this->output_size = dim_sta*dim_str;
  this->output_size = M * N;
}

//Load CSR
// Warning: this method must be called after a layer has been set with setLayer
void GustavsonsSpGEMMSDMemory::setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer,
                                                       metadata_address_t KN_metadata_id, metadata_address_t KN_metadata_pointer) {
  this->MK_col_id = MK_metadata_id;
  this->MK_row_pointer = MK_metadata_pointer;
  this->KN_col_id = KN_metadata_id;
  this->KN_row_pointer = KN_metadata_pointer;
  //Calculating number of nonzeros
  this->MK_number_nnz = MK_row_pointer[K];
  this->metadata_loaded = true;

  // Initialize index variables according to the first non-empty row
  for (int i = 0; i < this->M; i++) {
    if (MK_row_pointer[i] != MK_row_pointer[i + 1]) {
      this->current_MK_row_pointer = i;
      break;
    }
  }
}

void GustavsonsSpGEMMSDMemory::cycle() {
  // std::cout<<"Hello, its me";
  //Here MK(sparse) matrix is stationary and KN(dense) matrix is streaming
  //Sending input data over read_connection
  assert(this->layer_loaded);              // Layer has been loaded
  assert(this->metadata_loaded);           //Metadata for sparsity has been loaded
  std::vector<DataPackage*> data_to_send;  //Input and weight temporal storage
  //std::vector<DataPackage*> psum_to_send; // psum temporal storage
  this->local_cycle += 1;
  this->sdmemoryStats.total_cycles++;  //To track information
  while (mem.get_read_buffer_size() > 0) {
    DataPackage* pck = mem.get_read_buffer_front();
    mem.pop_read_buffer();

    if ((current_state == DIST_STR_MATRIX) || (current_state == WAITING_FOR_NEXT_STA_ITER)) {
      buffer_sync[pck->get_unicast_dest()].push(pck);  //To hidde the memory latency and send later all the elements at the same time
    } else {
      this->sendPackageToInputFifos(pck);  //This is for STA data
    }
  }

  //Processing write memory requests
  while (mem.get_write_buffer_size() > 0) {
    DataPackage* pck = mem.get_write_buffer_front();
    mem.pop_write_buffer();

    data_t data = pck->get_data();
    uint64_t addr = pck->get_address();
    addr = addr - this->output_dram_location;  //To access to the array. If we remove the array feature this is no longer necessary
    addr = addr / this->data_width;
    this->output_address[addr] = data;
    delete pck;
  }

  // If the GEMM has been already completed, then skip the rest of the steps (just wait until write_quee_ is completely empty)
  if (this->execution_finished)
    return;

  if (mem.get_read_buffer_size() == 0) {

    if (current_state == CONFIGURING) {
      //Initialize these for the first time
      this->n_str_data_received = 0;
      this->n_str_data_sent = 0;
      this->current_KN = 0;

      Tile* tile1 = new Tile(1, 1, 1, this->num_ms, 1, 1, 1, 1, false);
      this->tile = tile1;
      this->multiplier_network->resetSignals();
      //this->reduce_network->resetSignals();
      this->multiplier_network->configureSignals(tile1, this->dnn_layer, this->num_ms, 1, PSUM_GENERATION_AND_MERGE);  //TODO double check
      this->reduce_network->resetSignals();
      this->reduce_network->configureSignalsSortTree(SORT_TREE);
      //Cleaning the table to keep the index of each vector
      for (int i = 0; i < this->num_ms; i++) {
        ms_group[i] = -1;
      }

      pointer_current_memory = &swap_memory;
      pointer_next_memory = &intermediate_memory;  //where the first iteration will be stored
      std::cout << "Computing MK row " << current_MK_row_pointer << "/" << M << std::endl;

      //this->reduce_network->configureSignals(tile1, this->dnn_layer, this->num_ms, this->iter_K);
    }
    if (current_state == DIST_STA_MATRIX) {
      //Distribution of the stationary matrix
      //Sending unitcast message with each value in MK matrix
      multipliers_used = 0;
      this->sorting_iterations = ((MK_row_pointer[current_MK_row_pointer + 1] - MK_row_pointer[current_MK_row_pointer]) / this->num_ms) +
                                 (((MK_row_pointer[current_MK_row_pointer + 1] - MK_row_pointer[current_MK_row_pointer]) % this->num_ms) != 0);
      for (int i = 0; i < this->num_ms; i++) {
        //Accessing to the next value
        int row = current_MK_row_pointer;
        int col = MK_col_id[current_MK_col_id];
        multipliers_used++;
        //Sending package
        vnat_table[i] = col;  //To find out the col of mstrix KN.
        uint64_t new_addr = input_dram_location + current_MK_col_id * this->data_width;
        data_t data = 0.0;
        n_str_req_sent++;

        DataPackage* pck_to_send = new DataPackage(sizeof(data_t), data, WEIGHT, this->current_sorting_iteration, UNICAST, i, row, col);
        //	   std::cout << "[Cycle " << this->local_cycle << "] Sending data with value " << data << std::endl;
        //this->sendPackageToInputFifos(pck_to_send);
        doLoad(new_addr, pck_to_send);
        this->sdmemoryStats.n_SRAM_weight_reads++;
        //Update variables
        current_MK_col_id++;
        if (current_MK_col_id >= MK_row_pointer[current_MK_row_pointer + 1]) {
          do {
            current_MK_row_pointer += 1;
          } while (current_MK_row_pointer < M && MK_row_pointer[current_MK_row_pointer] == MK_row_pointer[current_MK_row_pointer + 1]);
          if (current_MK_row_pointer >= M) {
            //this->execution_finished=true; //The execution finishes here
            this->last_sta_iteration_completed = true;
          }

          break;  //The granularity of the iteration is one row
        }
      }
      this->STA_complete = true;

      //}

    }

    else if (current_state == DIST_STR_MATRIX) {  //Dense matrix
      bool found = false;
      n_str_req_sent = 0;
      for (int i = 0; i < multipliers_used; i++) {
        int row = vnat_table[i];  //Corresponds with the col of the value sta in the multiplier
        int length_row = KN_row_pointer[row + 1] - KN_row_pointer[row];
        // std::cout << "Comparing " << this->current_KN << " < " << length_row << std::endl;
        if (this->current_KN < length_row) {
          found = true;
          this->n_str_data_sent++;
          //Send STR value to this multiplier
          uint64_t new_addr = this->weight_dram_location + (KN_row_pointer[row] + this->current_KN) * this->data_width;
          data_t data = 0.0;
          DataPackage* pck_to_send = new DataPackage(sizeof(data_t), data, IACTIVATION, this->current_sorting_iteration, UNICAST, i, row,
                                                     KN_col_id[KN_row_pointer[row] + this->current_KN]);
          //    std::cout << "[Cycle " << this->local_cycle << "] Sending STREAMING data with value " << data << std::endl;
          //this->sendPackageToInputFifos(pck_to_send);
          doLoad(new_addr, pck_to_send);
          this->sdmemoryStats.n_SRAM_input_reads++;
          n_str_req_sent++;
        }
      }
      if (!found) {
        STR_complete = true;
        this->current_sorting_iteration += 1;

        // If we haven't read any data along the first iteration, then there is no STR data matching
        // the current STA row. Thus, we can skip next phases and go directly to the next STA row
        if (current_KN == 0) {
          this->sort_up_received_first_value = true;
        }
      }

      this->current_KN += 1;
    }

    else if (current_state == WAITING_FOR_NEXT_STA_ITER) {
      for (int i = 0; i < multipliers_used; i++) {
        if (buffer_sync[i].size() > 0) {
          DataPackage* pck_to_send = buffer_sync[i].front();
          buffer_sync[i].pop();
          this->sendPackageToInputFifos(pck_to_send);
        }
      }
    }

    else if (current_state == CONFIGURING_SORTING_PSUMS_DOWN) {
      //	std::cout << "CONFIGURING_SORTING_PSUMS_DOWN" << std::endl;
      this->reduce_network->resetSignals();
      this->reduce_network->configureSignalsSortTree(SORT_TREE);
      this->multiplier_network->resetSignals();
      this->multiplier_network->configureSignals(this->tile, this->dnn_layer, this->num_ms, 1, FORWARDER);
      this->n_str_data_received = 0;
      if (!swap_memory_enabled) {
        pointer_current_memory = &intermediate_memory;
        pointer_next_memory = &swap_memory;
      }

      else {
        pointer_current_memory = &swap_memory;
        pointer_next_memory = &intermediate_memory;
      }
      this->sorting_iterations = pointer_current_memory->size() / this->num_ms + ((pointer_current_memory->size() % this->num_ms) != 0);
      if (this->current_sorting_iteration == this->sorting_iterations) {  //Only when the number of iterations accomplished
        this->current_sorting_iteration = 0;
      }

    }

    else if (current_state == SENDING_SORT_TREE_DOWN) {
      //assert(pointer_current_memory->size() <= this->num_ms); //TODO implement this situation
      bool found = false;
      int i = this->current_sorting_iteration * this->num_ms;  //Accessing to the appropiate index
      int j = 0;
      while ((j < this->num_ms) && (i < pointer_current_memory->size())) {  //Using a second index j makes the code more clear
        if ((*pointer_current_memory)[i].size() > 0) {                      //If there are elements in this group
          found = true;
          DataPackage* pck_stored = (*pointer_current_memory)[i].front();
          (*pointer_current_memory)[i].pop();
          this->sdmemoryStats.n_SRAM_psum_reads++;
          int destination = j;
          DataPackage* pck_to_send = new DataPackage(sizeof(data_t), pck_stored->get_data(), PSUM, this->current_sorting_iteration, UNICAST, destination,
                                                     pck_stored->getRow(), pck_stored->getCol());
          //      std::cout << "[Cycle " << this->local_cycle << "] Sending data ROW=" << pck_to_send->getRow() << " COL=" << pck_to_send->getCol()  << " Data=" << pck_to_send->get_data() << " Destination: " << destination << " Current_iter: " << this->current_sorting_iteration << std::endl;
          delete pck_stored;
          this->sendPackageToInputFifos(pck_to_send);
        }
        i++;
        j++;
      }

      //Updating variables
      if (!found) {  // Continue to the next row
        this->current_sorting_iteration++;
        if (this->current_sorting_iteration == this->sorting_iterations) {
          if (this->sorting_iterations == 1) {
            this->swap_memory_enabled = false;
          }
        }

        this->sort_down_iteration_finished = true;
      }
    }

  }  //End if there is no pending requests
  //Receiving output data from write_connection
  this->receive();
  if (!write_fifo->isEmpty()) {
    waiting_idle_cycles = 0;
    for (int i = 0; i < write_fifo->size(); i++) {
      DataPackage* pck_received = write_fifo->pop();

      if ((current_state == DIST_STR_MATRIX) || (current_state == WAITING_FOR_NEXT_STA_ITER) || (current_state == SENDING_SORT_TREE_DOWN) ||
          (current_state == RECEIVING_SORT_TREE_UP)) {
        this->sort_up_received_first_value = true;
        //  this->output_address[pck_received->getRow()*this->N+n_str_data_received] = pck_received->get_data();
        if (this->sorting_iterations > 1) {
          //If there are more than 1 iterations, we have to keep the data in the intermediate memory
          int group = pck_received->get_source();
          //std::cout << "Group: " << group << std::endl;
          while (pointer_next_memory->size() <= group) {  //We have to create the space if a new group of psums are coming
            std::queue<DataPackage*> new_group;
            pointer_next_memory->push_back(new_group);
          }

          //Adding the element
          (*pointer_next_memory)[group].push(pck_received);
          this->sdmemoryStats.n_SRAM_psum_writes++;
          //std::cout << "Writing element into the intermediate memory" << std::endl;
        } else {
          //std::cout << "Writing element into the array with position " << n_values_stored << std::endl;
          std::size_t new_addr = this->output_dram_location + (pck_received->getRow() * this->N + pck_received->getCol()) * this->data_width;
          //std::cout << "Writing element in address " << new_addr << std::endl;
          //this->output_address[this->n_values_stored]=pck_received->get_data();
          pck_received->set_address(new_addr);

          // note: comment this store to hide write latency, but simulation won't return a result file
          doStore(new_addr, pck_received);
          this->sdmemoryStats.n_DRAM_psum_writes++;
          n_values_stored++;
          //delete pck_received;
        }
        this->n_str_data_received++;
      }
    }

  }  //End write_fifo

  else {  //If nothing is received
    waiting_idle_cycles++;
    if (((current_state == RECEIVING_SORT_TREE_UP) || (current_state == WAITING_FOR_NEXT_STA_ITER)) && this->sort_up_received_first_value &&
        (mem.get_read_buffer_size() == 0) && (waiting_idle_cycles < 800)) {
      //std::cout << "Closing this iteration" << std::endl;
      this->sort_up_received_first_value = false;
      if (this->last_sta_iteration_completed && (this->sorting_iterations == 1)) {  //If the last iteration has been streamed down before
        this->execution_finished = true;
        std::cout << "The execution has finished" << std::endl;
        this->sort_up_iteration_finished = true;
      }

      else {  //Switch to send more data to be merged or go to configuring

        if (this->current_state == WAITING_FOR_NEXT_STA_ITER) {
          this->STR_complete = true;
          //std::cout << "Completing the iteration" << std::endl;
        } else {
          this->sort_up_iteration_finished = true;
          if (this->current_sorting_iteration == this->sorting_iterations) {
            this->current_sorting_iteration = 0;
            if (this->sorting_iterations > 1) {
              this->swap_memory_enabled = !this->swap_memory_enabled;
              //Cleaning the current source memory which will be destination in the next iteration
              pointer_current_memory->clear();
            }

            //else {
            //    this->swap_memory_enabled = false;
            //}
          }

          //else {
          //    this->swap_memory_enabled = false;
          //}
        }
      }
    }
  }

  if ((current_state == CONFIGURING_SORTING_PSUMS_DOWN) || (current_state == SENDING_SORT_TREE_DOWN) || (current_state == RECEIVING_SORT_TREE_UP)) {
    this->sdmemoryStats.n_cycles_merging++;
  }

  else {
    this->sdmemoryStats.n_cycles_multiplying++;
  }

  //Transitions
  if ((current_state == CONFIGURING) && ((mem.get_read_buffer_size() == 0))) {
    current_state = DIST_STA_MATRIX;
    //std::cout << "Controller transiting from CONFIGURING to DIST_STA_MATRIX" << std::endl;
  }

  else if ((current_state == DIST_STA_MATRIX)) {
    if (STA_complete) {
      //std::cout << "Controller transiting from DIST_STA_MATRIX to DIST_STR_MATRIX" << std::endl;
      current_state = DIST_STR_MATRIX;
      STA_complete = false;
    }
  }

  else if ((current_state == DIST_STR_MATRIX)) {
    if (STR_complete) {
      //std::cout << "Controller transiting from DIST_STR_MATRIX to WAITING_FOR_NEXT_STA_ITER" << std::endl;
      this->current_state = WAITING_FOR_NEXT_STA_ITER;
      this->STR_complete = false;
    }
  }

  else if ((current_state == WAITING_FOR_NEXT_STA_ITER) && ((mem.get_read_buffer_size() == 0))) {
    if (STR_complete) {
      //std::cout << "Running from WAITING FOR NEXT STA ITER" << std::endl;
      STR_complete = false;
      //std::cout << "current_sorting_iteration: " << this->current_sorting_iteration << std::endl;
      //std::cout << "Sorting iterations: " << this->sorting_iterations << std::endl;;
      if (this->current_sorting_iteration < this->sorting_iterations) {
        current_state = CONFIGURING;
        //std::cout << "Controller transiting from WAITING_NEXT_STA_ITER to CONFIGURING" << std::endl;

      }

      else if (this->current_sorting_iteration == this->sorting_iterations) {
        //std::cout << "Simulated Iteration completed" << std::endl;
        if (this->sorting_iterations == 1) {
          this->current_state = CONFIGURING;
          //std::cout << "Controller transiting from WAITING_NEXT_STA_ITER to CONFIGURING" << std::endl;
          std::cout << "The row has been completed" << std::endl;

          this->current_sorting_iteration = 0;
          //std::cout << "GOING TO CONFIGURING" << std::endl;
        }

        else {
          std::cout << "The row is going to be sorted" << std::endl;
          this->current_state = CONFIGURING_SORTING_PSUMS_DOWN;
          this->current_sorting_iteration = 0;
          //std::cout << "Controller transiting from WAITING_NEXT_STA_ITER to CONFIGURING_SORTING_PSUMS_DOWN" << std::endl;

          // std::cout << "GOING TO CONFIGURING_SORT_TREE_DOWN" << std::endl;
        }
      }
    }

  }

  else if (current_state == CONFIGURING_SORTING_PSUMS_DOWN) {
    this->current_state = SENDING_SORT_TREE_DOWN;
    //std::cout << "Controller transiting from CONFIGURING_SORTING_PSUMS_DOWN to SENDING_SORT_TREE_DOWN" << std::endl;

  }

  else if (current_state == SENDING_SORT_TREE_DOWN) {
    if (this->sort_down_iteration_finished) {
      this->sort_down_iteration_finished = false;
      this->current_state = RECEIVING_SORT_TREE_UP;
      this->sort_up_received_first_value = false;
      // std::cout << "Controller transiting from SENDING_SORT_TREE_DOWN to RECEIVING_SORT_TREE_UP" << std::endl;
    }
  }

  else if (current_state == RECEIVING_SORT_TREE_UP) {
    if (this->sort_up_iteration_finished) {
      //std::cout << "Sort_up_iteration has finished" << std::endl;
      if (this->sorting_iterations > 1) {  //If there are more iterations, we still have to merge the next group of psums
        this->current_state = CONFIGURING_SORTING_PSUMS_DOWN;
        //std::cout << "Controller transiting from RECEIVING_SORT_TREE_UP to CONFIGURING_SORTING_PSUMS_DOWN" << std::endl;

        //	std::cout << "Getting back to CONFIGURING_PSUM_DOWN" << std::endl;
        //this->sort_up_received_first_value=false;
      }

      else {  //Otherwise we go to the next row
        this->current_state = CONFIGURING;
        std::cout << "A new row has been completed" << std::endl;
        //std::cout << "Controller transiting from RECEIVING_SORT_TREE_UP to CONFIGURING" << std::endl;
      }
      this->sort_up_iteration_finished = false;
    }
  }

  //else if(current_state==WAITING_FOR_NEXT_STA_ITER) {

  //}  //This state is modified when receiving data

  else if (current_state == ALL_DATA_SENT) {

    //	if(current_M>=this->M) {
    //Calculating sparsity values  and some final stats
    std::size_t sta_size = this->M * this->K;
    std::size_t sta_zeros = sta_size - this->n_ones_sta_matrix;
    std::size_t str_zeros = 0;
    sdmemoryStats.sta_sparsity = (counter_t)((100 * sta_zeros) / sta_size);
    sdmemoryStats.str_sparsity = (counter_t)0;
    this->sdmemoryStats.n_sta_vectors_at_once_avg = this->sdmemoryStats.n_sta_vectors_at_once_avg / this->sdmemoryStats.n_reconfigurations;
    current_state = ALL_DATA_SENT;
  }

  this->send();
}

bool GustavsonsSpGEMMSDMemory::isExecutionFinished() {
  return ((this->execution_finished) && (mem.get_write_buffer_size() == 0));
}

/* The traffic generation algorithm generates a package that contains a destination for all the ms. We have to divide it into smaller groups of ms since they are divided into several ports */
void GustavsonsSpGEMMSDMemory::sendPackageToInputFifos(DataPackage* pck) {
  // BROADCAST PACKAGE
  if (pck->isBroadcast()) {
    //Send to all the ports with the flag broadcast enabled
    for (int i = 0; i < (this->n_read_ports); i++) {
      //Creating a replica of the package to be sent to each port
      DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), pck->get_source(), BROADCAST, pck->getRow(),
                                             pck->getCol());  //Size, data, data_type, source (port in this case), BROADCAST
      //Sending the replica to the suitable fifo that correspond with the port
      if (pck->get_data_type() == PSUM) {  //Actually a PSUM cannot be broadcast. But we put this for compatibility
        psum_fifos[i]->push(pck_new);
      } else {  //INPUT OR WEIGHT
        //Seting iteration of the package
        pck_new->setIterationK(pck->getIterationK());  //Used to avoid sending packages from a certain iteration without performing the previous.
        input_fifos[i]->push(pck_new);
      }
    }
  }

  // UNICAST PACKAGE
  else if (pck->isUnicast()) {
    //We only have to send the weight to one port and change the destination to adapt it to the subgroup
    std::size_t dest = pck->get_unicast_dest();  //This is according to ALL the mswitches.
    std::size_t input_port = dest / this->ms_size_per_input_port;
    std::size_t local_dest = dest % this->ms_size_per_input_port;
    //Creating the package
    DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), pck->get_source(), UNICAST, local_dest,
                                           pck->getRow(), pck->getCol());  //size, data, type, source (port), UNICAST, dest_local
    //Sending to the fifo corresponding with port input_port
    if (pck->get_data_type() == PSUM) {  //Actually a PSUM cannot be broadcast. But we put this for compatibility
      psum_fifos[input_port]->push(pck_new);
      //std::cout << "Sending to the port " << input_port << std::endl;
    } else {  //INPUT OR WEIGHT
      input_fifos[input_port]->push(pck_new);
      pck_new->setIterationK(pck->getIterationK());
    }

  }

  //MULTICAST PACKAGE
  else {                                  //The package is multicast and then we have to send the package to several ports
    const bool* dest = pck->get_dests();  //One position for mswitch in all the msarray
    bool thereis_receiver;
    for (int i = 0; i < (this->n_read_ports); i++) {  //Checking each port with size this->ms_size_per_input_port each. Total=ms_size
      std::size_t port_index = i * this->ms_size_per_input_port;
      thereis_receiver = false;                                   // To know at the end if the group
      bool* local_dest = new bool[this->ms_size_per_input_port];  //Local destination array for the subtree corresponding with the port i
      for (int j = 0; j < this->ms_size_per_input_port; j++) {    //For each ms in the group of the port i
        local_dest[j] = dest[port_index + j];                     //Copying the subarray
        if (local_dest[j] == true) {
          thereis_receiver = true;  // To avoid iterating again to know whether the data have to be sent to the port or not.
        }
      }

      if (thereis_receiver) {  //If this port have at least one ms to true then we send the data to this port i
        DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), pck->get_source(), MULTICAST, local_dest,
                                               this->ms_size_per_input_port, pck->getRow(), pck->getCol());
        if (pck->get_data_type() == PSUM) {
          psum_fifos[i]->push(pck_new);
        }

        else {
          pck_new->setIterationK(pck->getIterationK());
          input_fifos[i]->push(pck_new);
        }
      } else {
        delete[] local_dest;  //If this vector is not sent we remove it.
      }
    }
  }

  delete pck;  // We have created replicas of the package for the ports needed so we can delete this
}

void GustavsonsSpGEMMSDMemory::send() {
  //Iterating over each port and if there is data in its fifo we send it. We give priority to the psums

  for (int i = 0; i < (this->n_read_ports); i++) {
    std::vector<DataPackage*> pck_to_send;
    if (!this->psum_fifos[i]->isEmpty()) {  //If there is something we may send data though the connection
      DataPackage* pck = psum_fifos[i]->pop();
#ifdef DEBUG_MEM_INPUT
      std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a psum through input port " << i << std::endl;
#endif
      pck_to_send.push_back(pck);
      this->sdmemoryStats.n_SRAM_read_ports_psums_use[i]++;  //To track information
      //Sending to the connection
      this->read_connections[i]->send(pck_to_send);
    }
    //If psums fifo is empty then input fifo is checked. If psum is not empty then else do not compute. Important this ELSE to give priority to the psums and do not send more than 1 pck
    else if (!this->input_fifos[i]->isEmpty()) {
      //If the package belongs to a certain k iteration but the previous k-1 iteration has not finished the package is not sent
      DataPackage* pck = input_fifos[i]->front();  //Front because we are not sure if we have to send it.

      if (pck->get_data_type() == WEIGHT) {
        this->sdmemoryStats.n_SRAM_read_ports_weights_use[i]++;  //To track information
#ifdef DEBUG_MEM_INPUT
        std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a WEIGHT through input port " << i << std::endl;
#endif
      } else {
        this->sdmemoryStats.n_SRAM_read_ports_inputs_use[i]++;  //To track information
#ifdef DEBUG_MEM_INPUT
        std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending an INPUT ACTIVATION through input port " << i << std::endl;
#endif
      }
      pck_to_send.push_back(pck);                    //storing into the vector data type structure used in class Connection
      this->read_connections[i]->send(pck_to_send);  //Sending the input or weight through the connection
      input_fifos[i]->pop();                         //pulling from fifo
    }
  }
}

//TODO Remove this connection
void GustavsonsSpGEMMSDMemory::receive() {  //TODO control if there is no space in queue
  if (this->write_connection->existPendingData()) {
    std::vector<DataPackage*> data_received = write_connection->receive();
    for (int i = 0; i < data_received.size(); i++) {
      write_fifo->push(data_received[i]);
    }
  }
  for (int i = 0; i < write_port_connections.size(); i++) {  //For every write port
    if (write_port_connections[i]->existPendingData()) {
      std::vector<DataPackage*> data_received = write_port_connections[i]->receive();
      for (int i = 0; i < data_received.size(); i++) {
        write_fifo->push(data_received[i]);
      }
    }
  }
}

void GustavsonsSpGEMMSDMemory::printStats(std::ofstream& out, std::size_t indent) {
  out << ind(indent) << "\"SDMemoryStats\" : {" << std::endl;  //TODO put ID
  this->sdmemoryStats.print(out, indent + IND_SIZE);
  out << ind(indent) << "}";  //Take care. Do not print endl here. This is parent responsability
}

void GustavsonsSpGEMMSDMemory::printEnergy(std::ofstream& out, std::size_t indent) {
  /*
        This component prints:
            - The number of SRAM reads
            - The number of SRAM writes

        Note that the number of times that each port is used is not shown. This is so because the use of those wires are
        taken into account in the CollectionBus and in the DSNetworkTop
   */

  counter_t reads = this->sdmemoryStats.n_SRAM_weight_reads + this->sdmemoryStats.n_SRAM_input_reads + this->sdmemoryStats.n_SRAM_psum_reads;
  counter_t writes = this->sdmemoryStats.n_SRAM_psum_writes;
  out << ind(indent) << "GLOBALBUFFER READ=" << reads;  //Same line
  out << ind(indent) << " WRITE=" << writes << std::endl;
}

bool GustavsonsSpGEMMSDMemory::doLoad(uint64_t addr, DataPackage* data_package) {
  mem.load(addr, data_package);
  return 1;
}

bool GustavsonsSpGEMMSDMemory::doStore(uint64_t addr, DataPackage* data_package) {
  mem.store(addr, data_package);
  return 1;
}
