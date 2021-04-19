
#include "SnapeaSDMemory.h"
#include <assert.h>
#include <iostream>
#include "utility.h"
#include <math.h>

SnapeaSDMemory::SnapeaSDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection) : MemoryController(id, name) {
	    this->write_connection = write_connection;
    //Collecting parameters from the configuration file
    this->num_ms = stonne_cfg.m_MSNetworkCfg.ms_size;  //Used to send data
    this->n_read_ports=stonne_cfg.m_SDMemoryCfg.n_read_ports;
    this->n_write_ports=stonne_cfg.m_SDMemoryCfg.n_write_ports;
    this->write_buffer_capacity=stonne_cfg.m_SDMemoryCfg.write_buffer_capacity;
    this->port_width=stonne_cfg.m_SDMemoryCfg.port_width;
    //End collecting parameters from the configuration file
    //Initializing parameters
    this->ms_size_per_input_port = this->num_ms / this->n_read_ports;
    this->write_fifo = new Fifo(write_buffer_capacity);
    for(int i=0; i<this->n_read_ports; i++) {
        Fifo* read_fi = new Fifo(this->write_buffer_capacity);
        Fifo* psum_fi = new Fifo(this->write_buffer_capacity);
        input_fifos.push_back(read_fi);
        psum_fifos.push_back(psum_fi);
        this->sdmemoryStats.n_SRAM_read_ports_weights_use.push_back(0);  //To track information
        this->sdmemoryStats.n_SRAM_read_ports_inputs_use.push_back(0);   //To track information
        this->sdmemoryStats.n_SRAM_read_ports_psums_use.push_back(0);    //To track information
    }
    for(int i=0; i<this->n_write_ports; i++) {  //To track information
        this->sdmemoryStats.n_SRAM_write_ports_use.push_back(0);  //To track information
    }  //To track information
    this->configuration_done = false;
    this->stationary_distributed = false;
    this->stationary_finished = false;
    this->stream_finished = false;
    this->execution_finished = false;
    this->metadata_loaded = false;
    this->layer_loaded = false;
    this->local_cycle=0;
    this->str_current_index = 0;
    this->current_state = CONFIGURING;
    this->sta_counters_table = NULL;
    this->str_counters_table = NULL;
    this->current_output = 0;
    this->output_size = 0;
    this->sta_current_index_metadata = 0; //Stationary matrix current index (e.g., row in MK)
    this->sta_current_index_matrix = 0;
    this->sta_iter_completed = false;
    this->current_output_iteration = 0;
    this->output_size_iteration = 0;
    this->sta_current_j_metadata = 0;
    this->sta_last_j_metadata = 0;
    this->n_ones_sta_matrix=0;
    this->n_ones_str_matrix=0;
    this->current_M=0;
    this->current_N=0;
    this->current_K_nnz=0;
    this->STA_complete=false;

    }

SnapeaSDMemory::~SnapeaSDMemory() {
    delete write_fifo;
    //Deleting the input ports
    for(int i=0; i<this->n_read_ports; i++) {
        delete input_fifos[i];
        delete psum_fifos[i];
    }
    

}

void SnapeaSDMemory::setWriteConnections(std::vector<Connection*> write_port_connections) {
    this->write_port_connections=write_port_connections; //Copying all the poiners 
    //assert(this->write_port_connections.size()==this->n_write_ports); 
}

void SnapeaSDMemory::setReadConnections(std::vector<Connection*> read_connections) {
    assert(read_connections.size() == n_read_ports); //Checking that the number of input ports is valid.
    this->read_connections = read_connections; //Copying all the pointers
}

void SnapeaSDMemory::setLayer(DNNLayer* dnn_layer, address_t MK_address, address_t KN_address, address_t output_address, Dataflow dataflow) {
    this->dnn_layer = dnn_layer;
    //assert(this->dnn_layer->get_layer_type()==DenseGEMM);  // This controller only supports GEMM with one sparse and one dense
    //this->dataflow = dataflow; 

    this->output_address = output_address;
    this->layer_loaded = true;


    //Loading parameters according to the equivalence between CNN layer and GEMM. This is done
    //in this way to keep the same interface.
    this->M = this->dnn_layer->get_X();
    this->K = this->dnn_layer->get_S();   //Be careful. K in GEMMs (SIGMA taxonomy) is not the same as K in CNN taxonomy (number of filters)
    this->N = this->dnn_layer->get_K();  //In this case both parameters match each other.
    sdmemoryStats.dataflow=dataflow; 
    
    this->MK_address = MK_address;
    this->KN_address = KN_address;


  //  this->output_size = dim_sta*dim_str;
      this->output_size = M*N;


}


//Dense Tiles
void SnapeaSDMemory::setTile(Tile* current_tile)
{
    this->T_K = current_tile->get_T_S();
    this->T_N = current_tile->get_T_K();;
    this->iter_N = ceil(this->N/(float)T_N);            //Zero-remainder constraint
    this->iter_M = M;

    //Tile* tile1 = new Tile(1, 1, T_K, T_N, 1, 1, 1, 1, false);

    //Generating the signals for the reduceNetwork and configuring it. The asnet->configureSignals will call its corresponding compiler to generate the signals and allocate all of them
    //this->multiplier_network->configureSignals(tile1, this->dnn_layer, this->num_ms); //Calling the ART to configure the signals with them previously generated
    //Getting MN signals
    //this->reduce_network->configureSignals(tile1, this->dnn_layer, this->num_ms);
    this->sdmemoryStats.n_reconfigurations++;
    for(int i=0;i<T_N;i++) {
        this->vnat_table_iterm.push_back(0);
        this->vnat_table_itern.push_back(0);
    }

}


void SnapeaSDMemory::cycle() {
   // std::cout<<"Hello, its me";
    //Here MK(sparse) matrix is stationary and KN(dense) matrix is streaming
    //Sending input data over read_connection
    assert(this->layer_loaded);  // Layer has been loaded
    //assert(this->metadata_loaded); //Metadata for sparsity has been loaded
    std::vector<DataPackage*> data_to_send; //Input and weight temporal storage
    //std::vector<DataPackage*> psum_to_send; // psum temporal storage
    this->local_cycle+=1;
    this->sdmemoryStats.total_cycles++; //To track information
    if(current_state==CONFIGURING)
    {	//Initialize these for the first time
    	this->K_nnz=this->K;
    	this->STA_base = current_M;;			
    	this->iter_K = K_nnz/T_K + (K_nnz%T_K!=0);
	std::cout << "Starting to compute output row " << current_M << "/" << M << std::endl;
	
	 this->multiplier_network->resetSignals();
	 this->reduce_network->resetSignals();
	 this->tile_to_configure = new Tile(1, 1, T_K, T_N, 1, 1, 1, 1, false);
	this->multiplier_network->configureSignals(this->tile_to_configure, this->dnn_layer, this->num_ms, this->iter_K);
	this->reduce_network->configureSignals(this->tile_to_configure, this->dnn_layer, this->num_ms, this->iter_K);
	this->current_K_nnz = 0;
	STA_complete=false;
    }
    if(current_state == DIST_STA_MATRIX) {
       //Distribution of the stationary matrix
       unsigned int dest = 0; //MS destination
       unsigned int index_K = current_K_nnz*T_K;
           for(int j=0; j<this->T_K; j++) {
	       data_t data;//Accessing to memory
	       if(index_K<K_nnz)
	       {					//This may solve zero remainder constraint too
	       	     data = this->MK_address[current_M*this->K+this->snapea_filter_order[current_M*this->K+index_K]];
		     //data = this->MK_address[current_M*this->K + index_K]; 
	       	     sdmemoryStats.n_SRAM_weight_reads++;
	       	     this->n_ones_sta_matrix++;
	       }
	       else
	       {
	             data = 0.0;
	       }
               bool* destinations = new bool[this->num_ms];
               for(int i=0; i<this->num_ms; i++) {
                   destinations[i]=false;
               }
               for(int i=0;i<this->T_N;i++){
                   destinations[i*T_K+j]=true;
               }
            
	       DataPackage* pck_to_send = new DataPackage(sizeof(data_t), data, WEIGHT, 0, MULTICAST, destinations, this->num_ms);
	       this->sendPackageToInputFifos(pck_to_send);
	       index_K++;
	   }
       //}
       
       this->current_K_nnz+=1;
            if(this->current_K_nnz == this->iter_K) {
                this->current_K_nnz = 0;
                this->STA_complete=true;
                }
    
    }

    else if(current_state == DIST_STR_MATRIX) {//Dense matrix
       unsigned int dest=0;
       unsigned int index_N=current_N*T_N;
       for(int i=0;i<T_N;i++)
       {
        unsigned int index_K=current_K_nnz*T_K;
       	   for(int j=0;j<T_K;j++)
       	   {
       	        data_t data;//Accessing to memory
	       if(index_K<K_nnz)
	       {
		     if(index_N < N) {  //Zero-remainder constraint for N-dim
	       	         //data = this->KN_address[index_N*this->K+index_K]; //Row of the dense matrix is indexed by the col id of sparse matrix
			 data = this->KN_address[index_N*this->K+this->snapea_filter_order[current_M*this->K+index_K]];
			 sdmemoryStats.n_SRAM_input_reads++;

		     }
		     else {
                         data = 0.0;
		     }
	       }
	       else
	       {
	             data = 0.0;
	       }
	       
	       DataPackage* pck_to_send = new DataPackage(sizeof(data_t), data, IACTIVATION, 0, UNICAST, dest);
	       this->sendPackageToInputFifos(pck_to_send);
	       
       		index_K++;
       		dest++;
       	   }
       	   index_N++;
       }
      
       this->current_K_nnz+=1;
            if(this->current_K_nnz == this->iter_K) {
                this->current_K_nnz = 0;
                this->current_N+=1;
                if(this->current_N == iter_N) {
                    this->current_N = 0;
                    this->current_M+=1;
                    this->STA_complete=true;
                } //end iter_N
            } //end iter_K_nnz
       
    }
   //Receiving output data from write_connection
    this->receive();
    if(!write_fifo->isEmpty()) {
        //Index the data by using the VN Address Table and the VN id of the packages
        for(int i=0; i<write_fifo->size(); i++) {
            DataPackage* pck_received = write_fifo->pop();
            unsigned int vn = pck_received->get_vn();
            data_t data = pck_received->get_data();
	    //unsigned int addr_offset = vnat_table[vn]*T_N + vn; 
	    unsigned int addr_offset = vnat_table_iterm[vn]*N + vnat_table_itern[vn]*T_N+vn;
	    if((vnat_table_itern[vn]*T_N+vn) < N) { //Zero-remainder constraint
                this->output_address[addr_offset]=data; //ofmap or psum, it does not matter.
                this->sdmemoryStats.n_SRAM_psum_writes++; //To track information
            }

	    vnat_table_itern[vn]++;
	    if(vnat_table_itern[vn]==this->iter_N) {
		vnat_table_itern[vn]=0;
                vnat_table_iterm[vn]++;
            } 
            current_output++;
	    current_output_iteration++;
	    if((current_output % 100) == 0) {
                std::cout << "Output completed " << current_output << "/" << M*N << ")" << std::endl;
            }

	    int N_and_pad = iter_N*T_N; 
	    if(current_output == M*N_and_pad) {
	       execution_finished=true;
	    }
	    if((current_output_iteration % T_N) == 0) {
                //RESET SIGNALS TO CLEAN THE PIPELINE AND REMOVE TRASH VALUES
		this->multiplier_network->resetSignals();
                this->reduce_network->resetSignals();
                this->multiplier_network->configureSignals(this->tile_to_configure, this->dnn_layer, this->num_ms, this->iter_K);
                this->reduce_network->configureSignals(this->tile_to_configure, this->dnn_layer, this->num_ms, this->iter_K);
		this->current_K_nnz = 0;
                //this->current_N+=1;
                //if(this->current_N == iter_N) {
                //    this->current_N = 0;
                //    this->current_M+=1;
                //    this->STA_complete=true;
                //} //end iter_N
		for(int i=0; i<this->n_read_ports; i++) {
	            while(!input_fifos[i]->isEmpty()) {
                        input_fifos[i]->pop(); //cleaning
		    }
		}

	    } 
	    if(current_output_iteration==N_and_pad) { //Later this would be N*T_M?
                current_output_iteration = 0;
		if(current_state == WAITING_FOR_NEXT_STA_ITER) {
		    STA_complete=true;
		}
	    }
            delete pck_received; //Deleting the current package
            
        }
    }

    //Transitions
    if(current_state==CONFIGURING) {
        current_state=DIST_STA_MATRIX;
    }

    else if(current_state==DIST_STA_MATRIX) {
        if(STA_complete) {
            current_state=DIST_STR_MATRIX;
	    STA_complete=false; // To be used in the DIST_STR_MATRIX state
	}
    }

    else if(current_state==DIST_STR_MATRIX) {
	if(current_M>=this->iter_M)
	    current_state=ALL_DATA_SENT;
        
	else {
		if(this->STA_complete) { //Otherwise current_state keeps the value DIST_STR_MATRIX
	            current_state = WAITING_FOR_NEXT_STA_ITER;
		    STA_complete=false;
		}
        }
    }

    else if(current_state == WAITING_FOR_NEXT_STA_ITER) {
        if(STA_complete) {
            current_state=CONFIGURING;
	} 
    }


    else if(current_state==ALL_DATA_SENT) {
    
	    current_state = ALL_DATA_SENT;

    }

   




    this->send();
}

bool SnapeaSDMemory::isExecutionFinished() {
    return this->execution_finished;
}

/* The traffic generation algorithm generates a package that contains a destination for all the ms. We have to divide it into smaller groups of ms since they are divided into several ports */
void SnapeaSDMemory::sendPackageToInputFifos(DataPackage* pck) {
    // BROADCAST PACKAGE
    if(pck->isBroadcast()) {
        //Send to all the ports with the flag broadcast enabled
        for(int i=0; i<this->n_read_ports; i++) {
            //Creating a replica of the package to be sent to each port
            DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), i, BROADCAST); //Size, data, data_type, source (port in this case), BROADCAST
            //Sending the replica to the suitable fifo that correspond with the port
            if(pck->get_data_type() == PSUM) { //Actually a PSUM cannot be broadcast. But we put this for compatibility
                psum_fifos[i]->push(pck_new);
            }          
            else {  //INPUT OR WEIGHT
                //Seting iteration of the package
                pck_new->setIterationK(pck->getIterationK()); //Used to avoid sending packages from a certain iteration without performing the previous.
                input_fifos[i]->push(pck_new);
            }     
           
        }
    }

    // UNICAST PACKAGE
    else if(pck->isUnicast()) {
        //We only have to send the weight to one port and change the destination to adapt it to the subgroup
        unsigned int dest = pck->get_unicast_dest(); //This is according to ALL the mswitches. 
        unsigned int input_port = dest / this->ms_size_per_input_port;
        unsigned int local_dest = dest % this->ms_size_per_input_port;
        //Creating the package 
        DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), input_port, UNICAST, local_dest); //size, data, type, source (port), UNICAST, dest_local
        //Sending to the fifo corresponding with port input_port
        if(pck->get_data_type() == PSUM) { //Actually a PSUM cannot be broadcast. But we put this for compatibility
            psum_fifos[input_port]->push(pck_new);
        }          
        else {  //INPUT OR WEIGHT
            input_fifos[input_port]->push(pck_new);
            pck_new->setIterationK(pck->getIterationK());
        }

    }

    //MULTICAST PACKAGE 
    else { //The package is multicast and then we have to send the package to several ports
        const bool* dest = pck->get_dests();  //One position for mswitch in all the msarray
        bool thereis_receiver;
        for(int i=0; i<this->n_read_ports; i++) { //Checking each port with size this->ms_size_per_input_port each. Total=ms_size
            unsigned int port_index = i*this->ms_size_per_input_port;
            thereis_receiver = false; // To know at the end if the group
            bool* local_dest = new bool[this->ms_size_per_input_port]; //Local destination array for the subtree corresponding with the port i
            for(int j=0; j<this->ms_size_per_input_port; j++) {  //For each ms in the group of the port i
                local_dest[j] = dest[port_index + j]; //Copying the subarray
                if(local_dest[j] == true) {
                    thereis_receiver=true; // To avoid iterating again to know whether the data have to be sent to the port or not.
                }
            }

            if(thereis_receiver) { //If this port have at least one ms to true then we send the data to this port i
                DataPackage* pck_new = new DataPackage(pck->get_size_package(), pck->get_data(), pck->get_data_type(), i, MULTICAST, local_dest, this->ms_size_per_input_port); 
                if(pck->get_data_type() == PSUM) {
                    psum_fifos[i]->push(pck_new);
                }
 
                else {
                    pck_new->setIterationK(pck->getIterationK());
                    input_fifos[i]->push(pck_new);
                    
                }
            }
            else {
                delete[] local_dest; //If this vector is not sent we remove it.
            }
        }
    }

    delete pck; // We have created replicas of the package for the ports needed so we can delete this
} 

void SnapeaSDMemory::send() {
    //Iterating over each port and if there is data in its fifo we send it. We give priority to the psums

    for(int i=0; i<this->n_read_ports; i++) {
        std::vector<DataPackage*> pck_to_send; 
        if(!this->psum_fifos[i]->isEmpty()) { //If there is something we may send data though the connection
            DataPackage* pck = psum_fifos[i]->pop();
#ifdef DEBUG_MEM_INPUT
            std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a psum through input port " << i  << std::endl;
#endif
            pck_to_send.push_back(pck);
            this->sdmemoryStats.n_SRAM_read_ports_psums_use[i]++; //To track information
            //Sending to the connection
            this->read_connections[i]->send(pck_to_send);
        }
        //If psums fifo is empty then input fifo is checked. If psum is not empty then else do not compute. Important this ELSE to give priority to the psums and do not send more than 1 pck
        else if(!this->input_fifos[i]->isEmpty()) {
            //If the package belongs to a certain k iteration but the previous k-1 iteration has not finished the package is not sent
            DataPackage* pck = input_fifos[i]->front(); //Front because we are not sure if we have to send it. 
           
            if(pck->get_data_type()==WEIGHT) {
                this->sdmemoryStats.n_SRAM_read_ports_weights_use[i]++; //To track information
#ifdef DEBUG_MEM_INPUT
                std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a WEIGHT through input port " << i << std::endl;
#endif
            }  
            else {
                this->sdmemoryStats.n_SRAM_read_ports_inputs_use[i]++; //To track information
#ifdef DEBUG_MEM_INPUT
                std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending an INPUT ACTIVATION through input port " << i << std::endl;
#endif
            }
                pck_to_send.push_back(pck); //storing into the vector data type structure used in class Connection 
                this->read_connections[i]->send(pck_to_send); //Sending the input or weight through the connection
            input_fifos[i]->pop(); //pulling from fifo
            

        }
            
    }


}

//TODO Remove this connection
void SnapeaSDMemory::receive() { //TODO control if there is no space in queue
    if(this->write_connection->existPendingData()) {
        std::vector<DataPackage*> data_received = write_connection->receive();
        for(int i=0; i<data_received.size(); i++) {
            write_fifo->push(data_received[i]);
        }
    }
    for(int i=0; i<write_port_connections.size(); i++) { //For every write port
        if(write_port_connections[i]->existPendingData()) {
            std::vector<DataPackage*> data_received = write_port_connections[i]->receive();
             for(int i=0; i<data_received.size(); i++) {
                 write_fifo->push(data_received[i]);
             }
        }    
    }
}

void SnapeaSDMemory::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"SDMemoryStats\" : {" << std::endl; //TODO put ID
    this->sdmemoryStats.print(out, indent+IND_SIZE);
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void SnapeaSDMemory::printEnergy(std::ofstream& out, unsigned int indent) {
    /*
        This component prints:
            - The number of SRAM reads
            - The number of SRAM writes

        Note that the number of times that each port is used is not shown. This is so because the use of those wires are
        taken into account in the CollectionBus and in the DSNetworkTop
   */

   counter_t reads = this->sdmemoryStats.n_SRAM_weight_reads + this->sdmemoryStats.n_SRAM_input_reads + this->sdmemoryStats.n_SRAM_psum_reads;
   counter_t writes = this->sdmemoryStats.n_SRAM_psum_writes;
   out << ind(indent) << "GLOBALBUFFER READ=" << reads; //Same line
   out << ind(indent) << " WRITE=" << writes << std::endl;
        
}

