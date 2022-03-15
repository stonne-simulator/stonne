
#include "SparseDenseSDMemory.h"
#include <assert.h>
#include <iostream>
#include "utility.h"
#include <math.h>

SparseDenseSDMemory::SparseDenseSDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection) : MemoryController(id, name) {
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

SparseDenseSDMemory::~SparseDenseSDMemory() {
    delete write_fifo;
    //Deleting the input ports
    for(int i=0; i<this->n_read_ports; i++) {
        delete input_fifos[i];
        delete psum_fifos[i];
    }
    
    if(this->layer_loaded) {
        delete[] sta_counters_table;
	delete[] str_counters_table;
    }

}

void SparseDenseSDMemory::setWriteConnections(std::vector<Connection*> write_port_connections) {
    this->write_port_connections=write_port_connections; //Copying all the poiners 
    //assert(this->write_port_connections.size()==this->n_write_ports); 
}

void SparseDenseSDMemory::setReadConnections(std::vector<Connection*> read_connections) {
    assert(read_connections.size() == n_read_ports); //Checking that the number of input ports is valid.
    this->read_connections = read_connections; //Copying all the pointers
}

void SparseDenseSDMemory::setLayer(DNNLayer* dnn_layer, address_t MK_address, address_t KN_address, address_t output_address, Dataflow dataflow) {
    this->dnn_layer = dnn_layer;
    assert(this->dnn_layer->get_layer_type()==SPARSE_DENSE);  // This controller only supports GEMM with one sparse and one dense
    //this->dataflow = dataflow; 

    this->output_address = output_address;
    this->layer_loaded = true;


    //Loading parameters according to the equivalence between CNN layer and GEMM. This is done
    //in this way to keep the same interface.
    this->M = this->dnn_layer->get_N();
    this->K = this->dnn_layer->get_C();   //Be careful. K in GEMMs (SIGMA taxonomy) is not the same as K in CNN taxonomy (number of filters)
    this->N = this->dnn_layer->get_K();  //In this case both parameters match each other.
    sdmemoryStats.dataflow=dataflow; 
    
    this->MK_address = MK_address;
    this->KN_address = KN_address;


  //  this->output_size = dim_sta*dim_str;
      this->output_size = M*N;
    this->sta_counters_table = new unsigned int[M*K];
    this->str_counters_table = new unsigned int[N*K];


}

//Load CSR
void SparseDenseSDMemory::setSparseMatrixMetadata(metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer) {
    this->MK_col_id = MK_metadata_id;
    this->MK_row_pointer = MK_metadata_pointer;
    this->metadata_loaded = true;

}

//Dense Tiles
void SparseDenseSDMemory::setDenseSpatialData(unsigned int T_N, unsigned int T_K)
{
    this->T_K = T_K;
    this->T_N = T_N;		
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


void SparseDenseSDMemory::cycle() {
   // std::cout<<"Hello, its me";
    //Here MK(sparse) matrix is stationary and KN(dense) matrix is streaming
    //Sending input data over read_connection
    assert(this->layer_loaded);  // Layer has been loaded
    assert(this->metadata_loaded); //Metadata for sparsity has been loaded
    std::vector<DataPackage*> data_to_send; //Input and weight temporal storage
    //std::vector<DataPackage*> psum_to_send; // psum temporal storage
    this->local_cycle+=1;
    this->sdmemoryStats.total_cycles++; //To track information
    if(current_state==CONFIGURING)
    {	//Initialize these for the first time
    	this->K_nnz=MK_row_pointer[current_M+1]-MK_row_pointer[current_M];
    	this->STA_base = MK_row_pointer[current_M];			
    	this->iter_K = K_nnz/T_K + (K_nnz%T_K!=0);
	//std::cout << "Starting to compute output row " << current_M << "/" << M << std::endl;
	
	 Tile* tile1 = new Tile(1, 1, T_K, T_N, 1, 1, 1, 1, false);
	 this->multiplier_network->resetSignals();
	 this->reduce_network->resetSignals();
	this->multiplier_network->configureSignals(tile1, this->dnn_layer, this->num_ms, this->iter_K);
	this->reduce_network->configureSignals(tile1, this->dnn_layer, this->num_ms, this->iter_K);
	this->current_K_nnz = 0;
	STA_complete=false;
    }
//    if(current_state==CONFIGURING) {   //If the architecture has not been configured
//        //int i=sta_current_index_metadata;  //Rows
//        
//        //Configure VN in terms of TK and TN
////        int i = current_M;
////	int j=0;  //Columns
////	int n_ms = 0; //Number of multipliers assigned
////	int n_current_cluster = 0;
//////	this->configurationVNs.clear();
////	this->vnat_table.clear();
//	
//	
///*
//	if(this->sta_current_j_metadata > 0)  { // We are managing one cluster with folding
//            n_ms++; //One for the psum
//	    j=this->sta_current_j_metadata;
//	    while((n_ms < this->num_ms) && (j < K)) { //TODO change MK if it is another dw
//                if(this->STA_metadata[i*STA_DIST_VECTOR + j*STA_DIST_ELEM]) { //If the bit is enabled in the stationary bitmap
//                    //Add to the cluster
//                    this->sta_counters_table[i*STA_DIST_VECTOR + j*STA_DIST_ELEM]=n_ms; //DEST
//                    n_ms++;
//                    n_current_cluster++;
//                }
//                
//		j++;

//	    }
//	    /*
//	    //Making sure that if there is next cluster, that cluster have size >=3
//	    if((j < K) && ((K-j) < 3)) {
//                int n_elements_to_make_cluster_3 = 3-(K-j);
//                j-=n_elements_to_make_cluster_3;
//                n_current_cluster-=n_elements_to_make_cluster_3;

//	    }



//            SparseVN VN(n_current_cluster, true); //Here folding is enabled and the SparseVN increments 1 to size
//            this->configurationVNs.push_back(VN);
//	    this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
//	    //Searching if there are more values in the next cluster. If not, update sta_last_j_metadata to K to indicate that in the next iteration the next sta dim must be evaluated
//	    int remaining_values = 0;
//	    for(int r=j; r<K; r++) {
//                if(this->STA_metadata[i*STA_DIST_VECTOR + r*STA_DIST_ELEM]) {
//                    remaining_values+=1;
//	        }
//	    }
//	    if(remaining_values > 0) {
//	        this->sta_last_j_metadata=j;
//	    }

//	    else {
//                this->sta_last_j_metadata = K;
//	    }



//	}

//	else { //Whole rows
//	    while((n_ms < this->num_ms) && (i*K+j < dim_sta*K)) { //TODO change MK if it is another dw
//                if(this->STA_metadata[i*STA_DIST_VECTOR + j*STA_DIST_ELEM]) { //If the bit is enabled in the stationary bitmap
//                    //Add to the cluster
//		    this->sta_counters_table[i*STA_DIST_VECTOR + j*STA_DIST_ELEM]=n_ms; //DEST
//		    n_ms++;
//		    n_current_cluster++; 
//	        }


//                j++; // Next elem in vector
//	        if(j==K) {
//		    //Change cluster since we change of vector
//                    j=0; //elem = 0
//		    i++; // Next vector
//		    if(n_current_cluster > 0) {
//                        //Creating the cluster for this row
//		        SparseVN VN(n_current_cluster, false);
//                        this->configurationVNs.push_back(VN); //Adding to the list 
//		        this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
//		        n_current_cluster = 0;

//		    }
//	        }

//	    }

//	    if((this->configurationVNs.size() > 0) && (j<K)) {
//                //Find if there is a last cluster
//		int remaining_values = 0;
//                for(int r=j; r<K; r++) {
//                    if(this->STA_metadata[i*STA_DIST_VECTOR + r*STA_DIST_ELEM]) {
//                        remaining_values+=1;
//                    }
//                }
//		//Its the last element
//                if(remaining_values == 0) {
//		    if(n_current_cluster > 0) {
//			SparseVN VN(n_current_cluster, false);
//                        this->configurationVNs.push_back(VN); //Adding to the list
//                        this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
//		    }

//                }
//	    
//	    }

//	    if(this->configurationVNs.size()==0) { //If any entire cluster fits, then folding is needed to manage this cluster
//		   /*
//		if((K-j) < 3) { //The next cluster must have cluster size greater or equal than 3
//                    int n_elements_to_make_cluster_3 = 3-(K-j);
//		    j-=n_elements_to_make_cluster_3;
//		    n_current_cluster-=n_elements_to_make_cluster_3;
//		}
//		
//	        SparseVN VN(n_current_cluster, false); //Here folding is still disabled as this is the first iteration
//	        this->configurationVNs.push_back(VN);
//		this->vnat_table.push_back(0); //Adding the current calculation (row or column) of this VN.
//		           //Searching if there are more values in the next cluster. If not, update sta_last_j_metadata to K to indicate that in the next iteration the next sta dim must be evaluated
//                int remaining_values = 0;
//                for(int r=j; r<K; r++) {
//                    if(this->STA_metadata[i*STA_DIST_VECTOR + r*STA_DIST_ELEM]) {
//                        remaining_values+=1;
//                    }
//                }
//                if(remaining_values > 0) {
//                    this->sta_last_j_metadata=j;
//                }

//                else {
//                    this->sta_last_j_metadata = K;
//                }

//              
//	    }

//	    else { //If there is at least one cluster, then all of them has size K and it is necessary to stream K
//		   //K elements
//		   this->sta_last_j_metadata=this->K;

//            }

//        } //end else whole rows

//*/

//	//Calculating the STR SOURCE TABLE with the indexes of each value
//	unsigned int source_id = 0;
//	for(int i=0; i<dim_str; i++) {
//            for(int j=0; j<K; j++) {
//                if(this->STR_metadata[i*STR_DIST_VECTOR+j*STR_DIST_ELEM]) {
//		    //std::cout << "Value (" << i << ", " << j << "): " << source_id << std::endl;
//                    this->str_counters_table[i*STR_DIST_VECTOR + j*STR_DIST_ELEM]=source_id;
//		    source_id++;
//		}
//	    }
//	}

//	this->n_ones_str_matrix = source_id;


//	//Once the VNs has been selected, lets configure the RN and MN
//        // Configuring the multiplier network
//	if(this->configurationVNs.size()==0) {
//            std::cout << "Cluster size exceeds the number of multipliers in row " << this->sta_current_index_metadata << std::endl;
//	    assert(false);
//	}
//	for(int i=0; i<this->configurationVNs.size(); i++) {
//            std::cout << "Found a VN of size " << this->configurationVNs[i].get_VN_Size() << std::endl;
//        }
//        this->sdmemoryStats.n_sta_vectors_at_once_avg+=this->configurationVNs.size(); //accumul
//	if(this->configurationVNs.size() > this->sdmemoryStats.n_sta_vectors_at_once_max) {
//            this->sdmemoryStats.n_sta_vectors_at_once_max = this->configurationVNs.size();
//	}
//	this->sdmemoryStats.n_reconfigurations++;
//	std::cout << "Configuring the Networks" << std::endl;
//	this->multiplier_network->resetSignals(); //Reseting the values to default
//	this->multiplier_network->configureSparseSignals(this->configurationVNs, this->dnn_layer, this->num_ms);
//	//Configuring the reduce network
//	this->reduce_network->resetSignals(); //Reseting the values to default
//	this->reduce_network->configureSparseSignals(this->configurationVNs, this->dnn_layer, this->num_ms);
//	std::cout << "End configuring" << std::endl;
//	//Number of psums to calculate in this iteration
//	this->output_size_iteration=this->configurationVNs.size()*this->dim_str;
//	

//    }

    if(current_state == DIST_STA_MATRIX) {
       //Distribution of the stationary matrix
       unsigned int dest = 0; //MS destination
       unsigned int index_K = current_K_nnz*T_K;
      
    
//       for(int i=0; i<this->configurationVNs.size(); i++) {
//	   int j=0;
//	   if(this->configurationVNs[i].getFolding()) {
//               j=1;
//	       dest++; //Avoid the one in charge of the psum
//	   }
           for(int j=0; j<this->T_K; j++) {
	       data_t data;//Accessing to memory
	       if(index_K<K_nnz)
	       {					//This may solve zero remainder constraint too
	       	     data = this->MK_address[STA_base+index_K]; 
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
	       	         data = this->KN_address[MK_col_id[STA_base+index_K]*N+index_N]; //Row of the dense matrix is indexed by the col id of sparse matrix
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
      /* int init_point_str = this->sta_current_j_metadata;
       int end_point_str = this->sta_last_j_metadata;
       if(this->sta_current_j_metadata > 0) { //If folding is enabled there is just a row on  fly
           assert(this->configurationVNs.size()==1);
           //send psum
	   unsigned int addr_offset = (sta_current_index_metadata)*OUT_DIST_VN + str_current_index*OUT_DIST_VN_ITERATION;
	   bool* destinations = new bool[this->num_ms];
           for(int i=0; i<this->num_ms; i++) {
               destinations[i]=false;
           }
	   destinations[0]=true;

	   data_t psum = this->output_address[addr_offset];  //Reading the current psum
	   DataPackage* pck = new DataPackage(sizeof(data_t), psum, PSUM,0, MULTICAST, destinations, this->num_ms);
           this->sdmemoryStats.n_SRAM_psum_reads++; //To track information
	   this->sendPackageToInputFifos(pck);
	   
       }
       for(int j=init_point_str; j<end_point_str; j++) {   //For each element in the current vector in the str matrix
         //Creating the bit vector for this value
	 bool* destinations = new bool[this->num_ms];
	 for(int i=0; i<this->num_ms; i++) {
             destinations[i]=false;
	 }

	 //Send the value i times with bit on bitmap enabled
	 //Searching for destinations for this elemeent
	 unsigned int first_sta_vector = sta_current_index_metadata;
	 unsigned int last_sta_vector = sta_current_index_metadata + this->configurationVNs.size();
         for(int i=first_sta_vector; i<last_sta_vector; i++) { //For each vector in the sta matrix could exist a value to send the str value
             if(STA_metadata[i*STA_DIST_VECTOR + j*STA_DIST_ELEM]) { //ADD Destination
		 unsigned int dest = sta_counters_table[i*STA_DIST_VECTOR + j*STA_DIST_ELEM];
                 destinations[dest]=true;
	     }
	 }

	 //Accessing to the element in the str matrix
	 data_t data;
	 if(STR_metadata[str_current_index*STR_DIST_VECTOR + j*STR_DIST_ELEM]) { 
	     unsigned int src = str_counters_table[str_current_index*STR_DIST_VECTOR + j*STR_DIST_ELEM];
	     data = STR_address[src];
	     sdmemoryStats.n_SRAM_input_reads++;
	 }

	 else {
             data=0.0; //If the STA matrix has a value then the STR matrix must be sent even if the value is 0
         }

	 //Creating the package
         DataPackage* pck = new DataPackage(sizeof(data_t), data,IACTIVATION,0, MULTICAST, destinations, this->num_ms);

	 this->sendPackageToInputFifos(pck);
       } 

       // str_current_index++;
    */   
        //Update dimensions
       
       this->current_K_nnz+=1;
            if(this->current_K_nnz == this->iter_K) {
                this->current_K_nnz = 0;
                this->current_N+=1;
                if(this->current_N == iter_N) {
                    this->current_N = 0;
                    this->current_M+=1;
                    this->STA_complete=true;
                    //if(current_M<iter_M){
                   // 	this->K_nnz=MK_row_pointer[current_M+1]-MK_row_pointer[current_M];	
                   // 	this->STA_base = MK_row_pointer[current_M];
		   //	std::cout << "The second value of iter_k is: " << this->iter_K << std::endl;

                    		
                    //if(this->current_M == iter_M) {
                     //                             this->STA_complete=true;

//                                            //Update K. Weight distribution neccesary first!
//                                                this->current_N = 0;
//                                                this->weights_distributed = true; //Avoid distribution
//                                                this->weights_finished=true;
//                                                //Current_K updated when weights are distributed
//                                                //Checking if all the inputs have been delivered ()
//                                                if(weights_finished) {
//                                                    this->input_finished=true;
//                                                }
                    //} //end iter_M
                } //end iter_N
            } //end iter_K_nnz
       
    }
/*
      //Update dimensions

       this->current_K_nnz+=1;
            if(this->current_K_nnz == this->iter_K) {
                this->current_K_nnz = 0;
                this->current_N+=1;
                if(this->current_N == iter_N) {
                    this->current_N = 0;
                    this->current_M+=1;
                    this->K_nnz=MK_row_pointer[current_M+1]-MK_row_pointer[current_M];		
    		    this->iter_K = K_nnz/T_K + (K_nnz%T_K!=0);
                    if(this->current_M == iter_M) {
//                                            //Update K. Weight distribution neccesary first!
//                                                this->current_N = 0;
//                                                this->weights_distributed = true; //Avoid distribution
//                                                this->weights_finished=true;
//                                                //Current_K updated when weights are distributed
//                                                //Checking if all the inputs have been delivered ()
//                                                if(weights_finished) {
//                                                    this->input_finished=true;
//                                                }
                    } //end iter_N
                } //end iter_N
            } //end iter_K_nnz
            
 */
    
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
                //this->clocked_op[addr_offset]=local_cycle;
                this->sdmemoryStats.n_SRAM_psum_writes++; //To track information
            }

	    vnat_table_itern[vn]++;
	    if(vnat_table_itern[vn]==this->iter_N) {
		vnat_table_itern[vn]=0;
                vnat_table_iterm[vn]++;
            } 
            current_output++;
	    //if((current_output % 1000) == 0) {
               // std::cout << "Output completed " << current_output << "/" << M*N << ")" << std::endl;
            // }

	    int N_and_pad = iter_N*T_N; 
	    if(current_output == M*N_and_pad) {
	       execution_finished=true;
	    }
	    current_output_iteration++;
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

    //else if(current_state==WAITING_FOR_NEXT_STA_ITER) {
        
    //}  //This state is modified when receiving data

    else if(current_state==ALL_DATA_SENT) {
    
//	this->str_current_index = 0;
//	this->sta_iter_completed=false;
//        if(this->configurationVNs.size()==1) {//If there is only one VN, then maybe foliding has been needed
//            this->sta_current_j_metadata=this->sta_last_j_metadata;
//	   // if(this->configurationVNs[0].getFolding()) {
//           //     this->sta_current_j_metadata-=1;
//	   // }

//	    if(this->sta_current_j_metadata == this->K) { //If this is the end of the cluster, it might start to the next 
//                this->sta_current_index_metadata+=1;
//		this->sta_current_j_metadata = 0;
//		std::cout << "STONNE: STA dimensions completed (" << this->sta_current_index_metadata << "/" << this->dim_sta << ")" << std::endl;


//            }
//	}

//	else {
//	    this->sta_current_index_metadata+=this->configurationVNs.size();
//	    std::cout << "STONNE: STA dimensions completed (" << this->sta_current_index_metadata << "/" << this->dim_sta << ")" << std::endl;
//	    this->sta_current_j_metadata = 0;
//	}
//	unsigned int total_size = 0;
//	for(int i=0; i<this->configurationVNs.size(); i++) {
//            total_size+=this->configurationVNs[i].get_VN_Size();
//	    if(this->configurationVNs[i].getFolding()) {
//                total_size-=1; //Sustract the -1 of the extra multiplier 
//	    }
//        }
//	this->sta_current_index_matrix+=total_size;

//	if(current_M>=this->M) {
	    //Calculating sparsity values  and some final stats
	    unsigned int sta_size = this->M*this->K;
	    unsigned int sta_zeros = sta_size - this->n_ones_sta_matrix;
	    unsigned int str_zeros = 0;
            sdmemoryStats.sta_sparsity=(counter_t)((100*sta_zeros) / sta_size);
	    sdmemoryStats.str_sparsity=(counter_t)0;
	    this->sdmemoryStats.n_sta_vectors_at_once_avg = this->sdmemoryStats.n_sta_vectors_at_once_avg / this->sdmemoryStats.n_reconfigurations;
	    current_state = ALL_DATA_SENT;

    }

   




    this->send();
}

bool SparseDenseSDMemory::isExecutionFinished() {
    return this->execution_finished;
}

/* The traffic generation algorithm generates a package that contains a destination for all the ms. We have to divide it into smaller groups of ms since they are divided into several ports */
void SparseDenseSDMemory::sendPackageToInputFifos(DataPackage* pck) {
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

void SparseDenseSDMemory::send() {
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
void SparseDenseSDMemory::receive() { //TODO control if there is no space in queue
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

void SparseDenseSDMemory::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"SDMemoryStats\" : {" << std::endl; //TODO put ID
    this->sdmemoryStats.print(out, indent+IND_SIZE);
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void SparseDenseSDMemory::printEnergy(std::ofstream& out, unsigned int indent) {
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

