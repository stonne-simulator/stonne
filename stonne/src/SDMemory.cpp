// Created by Francisco Munoz Martinez on 02/07/2019

#include "SDMemory.h"
#include <assert.h>
#include <iostream>
#include "utility.h"

//The memory is ordered setting the channels in consecutive words in memory

VNAT_Register::VNAT_Register(unsigned int VN, unsigned int addr, unsigned int N, unsigned int G, unsigned int K, unsigned int X, unsigned int Y,  unsigned int iter_N, unsigned int iter_G, unsigned int iter_K, unsigned int iter_X, unsigned int iter_Y, unsigned int iter_R, unsigned int iter_S, unsigned int iter_C, DNNLayer* dnn_layer, Tile* current_tile) {
        this->VN = VN;
        this->base_addr = addr; //This address is always fixed since is the one use to make the calculation of next address easier.
        this->addr = addr; //This addr change over time
        this->current_N = 0;
        this->current_G = 0;
        this->current_K = 0;
        this->current_X = 0;
        this->current_Y = 0;
        this->current_R = 0;
        this->current_S = 0;
        this->current_C = 0;
        this->iter_N = iter_N;
        this->iter_G = iter_G;
        this->iter_K = iter_K;
        this->iter_X = iter_X;
        this->iter_Y = iter_Y;
        this->iter_R = iter_R;
        this->iter_S = iter_S;
        this->iter_C = iter_C;
        this->n_psums=iter_R*iter_S*iter_C;
        this->current_psum=0;
        this->dnn_layer = dnn_layer;
        this->current_tile = current_tile;
}

//Return the offset from 0
unsigned int VNAT_Register::get_address() {
    return this->base_addr + addr;
}


//TODO revisar esta funcion 
void VNAT_Register::update() {
        this->current_S+=1;
        if(this->current_S == this->iter_S) {
            this->current_S = 0;
            this->current_R+=1;
            if(this->current_R == this->iter_R) {
                this->current_R = 0;
                this->current_C+=1;
                if(this->current_C == this->iter_C) {
                    this->current_C = 0;
                    this->current_Y+=1; //Updating cols
                    if(this->current_Y==this->iter_Y) {
                        //Updating X
                        this->current_Y=0; //Reset Y
                        this->current_X+=1;
                        // this->current_tile->get_T_X_()-1 because the address is already in next X iteration  (since it is consecutive to the previous Y)
                        if(this->current_X == this->iter_X) {
                        //If rows finished, updating next N batch
                            this->current_X = 0; //Updating X
                            this->current_K+=1;
                            if(this->current_K==this->iter_K) { //Go to next N
                                this->current_K = 0;
                                this->current_G+=1;
                                if(this->current_G==this->iter_G) {
                                    this->current_G=0;
                                    this->current_N++;
                                }
                                //assert(this->current_N > this->iter_N); 
                            } //end iter_K
                        } //end iter_X

                    } //end iter_Y 
                }  //end C
            } //end R
        } //End S
        unsigned index_N=current_N*this->current_tile->get_T_N();
        unsigned index_X=current_X*this->current_tile->get_T_X_();
        unsigned index_Y=current_Y*this->current_tile->get_T_Y_();
        unsigned index_K=current_K*this->current_tile->get_T_K();
        unsigned index_G=current_G*this->current_tile->get_T_G();
     
        this->addr = this->base_addr + (index_N)*dnn_layer->get_X_()*dnn_layer->get_Y_()*dnn_layer->get_K()*dnn_layer->get_G()*word_size + index_X*dnn_layer->get_Y_()*dnn_layer->get_K()*dnn_layer->get_G()*word_size + index_Y*dnn_layer->get_K()*dnn_layer->get_G()*word_size + index_G*dnn_layer->get_K()*word_size + index_K*word_size;
        //std::cout << "Address: " << this->addr << std::endl;
  
    
}

SDMemory::SDMemory(id_t id, std::string name, Config stonne_cfg, Connection* write_connection) : MemoryController(id, name) {
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
    this->weights_distributed = false;
    this->fw_link_enabled = false;
    this->weights_finished = false;
    this->input_finished = false;
    this->tile_loaded = false;
    this->execution_finished = false;
    this->current_output_pixel = 0;
    this->local_cycle=0;
    
    /* Now this is done by the bus
    //Creating the write ports
    for(int i=0; i<this->n_write_ports; i++) {
        Connection* connection = new Connection(this->n_write_ports); // TODO The capacity would be 1. Change this later
        write_port_connections.push_back(connection);
    }
    */

}

SDMemory::~SDMemory() {
    delete write_fifo;
    if(this->tile_loaded) {
        unsigned int num_vns = this->current_tile->get_Num_VNs();
        for(int i=0; i<num_vns; i++) {
            delete VNAT[i];
        }     
        delete[] VNAT;   
    }
    //Deleting the input ports
    for(int i=0; i<this->n_read_ports; i++) {
        delete input_fifos[i];
        delete psum_fifos[i];
    }

}

void SDMemory::setWriteConnections(std::vector<Connection*> write_port_connections) {
    this->write_port_connections=write_port_connections; //Copying all the poiners 
    //assert(this->write_port_connections.size()==this->n_write_ports); 
}

void SDMemory::setReadConnections(std::vector<Connection*> read_connections) {
    assert(read_connections.size() == n_read_ports); //Checking that the number of input ports is valid.
    this->read_connections = read_connections; //Copying all the pointers
}

void SDMemory::setTile(Tile* current_tile) {
        //Calculating th enumber of iterations
    //assert(this->write_port_connections.size()==this->n_write_ports);
    this->iter_R = dnn_layer->get_R() / current_tile->get_T_R(); //Control the number of R iterations in the MemoryController
    this->iter_S = dnn_layer->get_S() / current_tile->get_T_S(); //Control the number of S iterations in the MemoryController
    this->iter_C = dnn_layer->get_C() / current_tile->get_T_C(); //Control the number of C iterations in the MemoryController
    this->iter_N = dnn_layer->get_N() / current_tile->get_T_N(); //Control the number of N iterations in the MemoryController
    this->iter_G = dnn_layer->get_G() / current_tile->get_T_G(); //Control the number of G iterations in the MemoryController
    //std::cout << "iter G en inputs esssss " << iter_G << std::endl;
    this->iter_K = dnn_layer->get_K() / current_tile->get_T_K(); //Control the number of K iterations in the MemoryController
    this->iter_X = dnn_layer->get_X_() / current_tile->get_T_X_(); //Control the number of X iterations in the MemoryController
    this->iter_Y = dnn_layer->get_Y_() / current_tile->get_T_Y_();  //Control the number of Y iterations in the MemoryController
    unsigned int VNAT_iter_R = 1; //Control the number of R iterations in the VNAT (when writing into the memory). Only neccesary if the memory have to forward psms 
    unsigned int VNAT_iter_S = 1; //Control the number of S iterations in the VNAT (when writing into the memory). Only neccesary if the memory have to forward psms 
    unsigned int VNAT_iter_C = 1; //Control the number of C iterations in the VNAT (when writing into the memory). Only neccesary if the memory have to forward psms 
    if(current_tile->get_folding_enabled()) { //If folding is managed by the multipliers (forwarding enabled) then the memory has to be aware and store and send the partial sums to the multipliers
        VNAT_iter_R = this->iter_R;
        VNAT_iter_S = this->iter_S;
        VNAT_iter_C = this->iter_C;
    }
    std::cout << "dnn_layer_X_: " << dnn_layer->get_X_() << std::endl;
    std::cout << "current_tile_X: " << current_tile->get_T_X_() << std::endl;
    std::cout << "Iter_X=" << this->iter_X << std::endl;
    std::cout << "dnn_layer_Y_: " << dnn_layer->get_Y_() << std::endl;
    std::cout << "current_tile_y: " << current_tile->get_T_Y_() << std::endl;
    std::cout << "Iter_Y=" << iter_Y << std::endl;
    this->current_tile = current_tile;
    unsigned int num_vn = this->current_tile->get_Num_VNs();
    std::cout << "num vn = " << num_vn << std::endl;
    this->current_output_pixel = 0; //Setting the current pixels computed
    //The number of opixels to compute is iter_N*iter_K*iter_X*iter_Y * num_vn since each neuron is going to perform those iterations.
    // Notice this number might be different to the number of opixels in the dnn (it would be dnn_layer->get_K()*dnn_layer->get_X_()*dnn_layer->get_Y()
    // This is so because a DNN execution could need several tiles to complete.
    this->output_pixels_to_compute = this->iter_N*this->iter_G*this->iter_K*this->iter_X*this->iter_Y*VNAT_iter_R*VNAT_iter_S*VNAT_iter_C*num_vn;  
    //Number of output psums per each channel. Used to avoid sending packages of new k iterations if the previous have not been 
    //calculated yet;
    std::cout << "dnn_layer X: " << this->dnn_layer->get_X_() << std::endl; 
   
    this->output_psums_per_channel = this->dnn_layer->get_X_()*this->dnn_layer->get_Y_()*VNAT_iter_R*VNAT_iter_S*VNAT_iter_C; 
    std::cout << "psums to compute " << this->output_pixels_to_compute << std::endl;
    //Assigning to each VN an output initial address depending on N, K, X' and Y'
    this->VNAT = new VNAT_Register*[num_vn];
    for(unsigned n=0;n<this->current_tile->get_T_N(); n++) {
        for(unsigned g=0; g<this->current_tile->get_T_G(); g++) {
            for(unsigned k=0; k<this->current_tile->get_T_K(); k++) {
                for(unsigned x=0; x<this->current_tile->get_T_X_(); x++) {
                    for(unsigned y=0; y<this->current_tile->get_T_Y_();y++) {
                        unsigned int current_vn = n*this->current_tile->get_T_G()*this->current_tile->get_T_K()*this->current_tile->get_T_X_()*this->current_tile->get_T_Y_() + g*this->current_tile->get_T_K()*this->current_tile->get_T_X_()*this->current_tile->get_T_Y_() + k*this->current_tile->get_T_X_()*this->current_tile->get_T_Y_() + x*this->current_tile->get_T_Y_() + y;
                    
                        //The way of calculating the address is different since the data is storaged putting the different k channels consecutive in // memory
                        unsigned int addr_offset = n*this->dnn_layer->get_G()*this->dnn_layer->get_K()*this->dnn_layer->get_X_()*this->dnn_layer->get_Y_()*word_size + x*this->dnn_layer->get_Y_()*this->dnn_layer->get_G()*this->dnn_layer->get_K()*word_size + y*this->dnn_layer->get_G()*this->dnn_layer->get_K()*word_size + g*this->dnn_layer->get_K() + k*word_size;
                        assert(current_vn < num_vn); //Making sure it works
                        this->VNAT[current_vn]=new VNAT_Register(current_vn, addr_offset, n, g, k, x, y, iter_N, iter_G, iter_K, iter_X, iter_Y, VNAT_iter_R, VNAT_iter_S, VNAT_iter_C, this->dnn_layer, this->current_tile);
                    }
                }
            } //End K
        } //End G
    }
    this->tile_loaded = true;
}

void SDMemory::setLayer(DNNLayer* dnn_layer, address_t input_address, address_t filter_address, address_t output_address, Dataflow dataflow) {
    this->dnn_layer = dnn_layer;
    this->filter_address = filter_address;
    this->input_address = input_address;
    this->output_address = output_address;

    //Dataflow is always ignored here as the order of the loops is always the same when running a CNN
    //
    //i.e., you can change the dataflow in terms of tiling partition but not in terms of loop


    //Updating counters to track the progress
    this->current_N = 0;
    this->current_K = 0;
    this->current_G = 0;
    this->current_X = 0;
    this->current_Y = 0;
    this->current_R = 0;
    this->current_S = 0;
    this->current_C = 0;
    this->channel_filter_size = dnn_layer->get_R()*dnn_layer->get_S(); //R*S
    this->row_filter_size = dnn_layer->get_S()*dnn_layer->get_C();
    this->filter_size = dnn_layer->get_R()*dnn_layer->get_S()*dnn_layer->get_C(); //R*S*C
    this->group_size = dnn_layer->get_K()*dnn_layer->get_R()*dnn_layer->get_S()*dnn_layer->get_C(); //filter_size*K
    this->channel_input_size;
    this->row_input_size = dnn_layer->get_G()*dnn_layer->get_C()*dnn_layer->get_Y(); //Row input size = C*Y 
    this->input_size = dnn_layer->get_X()*dnn_layer->get_Y()*dnn_layer->get_C()*dnn_layer->get_G(); // ISize = X*Y*X
    this->channel_output_size = dnn_layer->get_X_()*dnn_layer->get_Y_();  //output channel size= X'*Y'
    this->row_output_size =  dnn_layer->get_G()*dnn_layer->get_K()*dnn_layer->get_Y_(); 
    this->output_size = dnn_layer->get_X_()*dnn_layer->get_Y_()*dnn_layer->get_G()*dnn_layer->get_K(); //Osize = X'*Y'*K
    this->output_pixels_to_compute = this->output_size*dnn_layer->get_N();
    //Configuring the controller
   /*
                    unsigned int filter_size = dnn_layer->get_R()*dnn_layer->get_S()*dnn_layer->get_C(); //R*S*C
                    unsigned int input_size = dnn_layer->get_X()*dnn_layer->get_Y()*dnn_layer*get_C(); // ISize = X*Y*X
                    unsigned int row_input_size = dnn_layer->get_C()*dnn_layer->get_Y(); //Row input size = C*Y
                    unsigned int output_size = dnn_layer->get_X_()*dnn_layer->get_Y_()*dnn_layer->get_K(); //Osize = X'*Y'*K
                    unsigned int channel_output_size = dnn_layer->get_X_()*dnn_layer->get_Y_();  //output channel size= X'*Y'
                    unsigned int row_output_size = dnn_layer->get_K()*dnn_layer->get_Y_(); 
                    unsigned int stride = dnn_layer->get_stride();
                    address_t current_filter_address = filter_address +  k*filter_size*word_size;
                    address_t current_input_address = input_address + n*input_size*word_size + x_*row_input_size*word_size*stride + 
                                                                     y_*dnn_layer->get_C()*word_size*stride;
                    address_t current_output_address = output_address + n*output_size*word_size + k*channel_output_size*word_size +
                                                                     x_*row_output_size*word_size +  y_*dnn_layer->get_K();
                    unsigned int current_T_R = current_tile->get_T_R();
                    unsigned int current_T_S = current_tile->get_T_S();
                    unsigned int current_T_C = current_tile->get_T_C();
                    unsigned int current_T_K = k;
                    unsigned int current_T_N = n;
                    unsigned int current_T_X_ = x_;
                    unsigned int current_T_Y_ = y_;

                    //Creating the current VN Register
                    VNRegister* vn_register = new VNRegister(VN, current_input_address, current_output_address, current_filter_address, 
                                             current_T_R, current_T_S, current_T_C, current_T_K, current_T_N, current_T_X_, current_T_Y_);
                    VNAT.push_back(vn_register);    
                    VN++; */
}
/*
TODO la idea de como implementarlo.
Quitar del VNAT todos los parametros current_T* y meterlos dentro de SDMemory ya que es el controlador quien va a llevar la cuenta del bucle 
principal. Cada ciclo se incrementara una variable u otra basado en el tile y en su tamano. Tambien, la direccion del input y output sera controlada
de manera global por el SDMemory. POr lo que hay que cambiar esto. 

Primero, si no esta activa la variable weight distributed quiere decir que hay que distribuir pesos. En dicho caso hacemos distincion entre
si la distribucion de pesos sera unicast o multicast. Dado que un mismo peso seria enviado N*X_*Y_ veces (ya qeu todas estas VN usarian el mismo
peso en una cierta posicion de MS), si N*X_*Y_ es 1 qquiere decir que los paquetes se envian en forma unicast, ya que no hhay reutilizacion de
los pesos espacialmentee. 

En ambos casos se recorre cada peso y se decide donde enviar. Si es unicast, simplemente se calcula el MS a enviar usando el parametro K y el desp
del peso. Si es multicast se crea un array bool que se enviara con el paquete del peso y se pondra a true en todos los MSs necesarios basandonso en 
N*X_*Y_ para cada K. Una vez sabeemos donde enviar, se direcciona la memoria para acceder al peso actual y se envia en un datapackage segun sea
unicast o multicast. 

Una vez enviados los pesos, se enviarian las neuronas correspondientes. Para ello, con la variable fw_link_enabled se decide cuantas neuronas hay
que enviar. Simplemente en cada ciclo se calcula este numero. Basicamente VN size = T_X*T_Y. Se recorre cada neurona y se decide si se manda a
un unico VN o a mas basandonos en T_X_ y T_Y_ y el stride. Para cada neurona se crea un bool gual que antes y se pone a true los correspondientes MS.
Nota: darse cuenta de que si T_K>1 todos los T_K deberan recibir la misma distribucion de neuronas, por lo uqe habra un true en cada T_K en el 
array de bool de destinos. 

*/

//Read data from memory


void SDMemory::cycle() {
    //Sending input data over read_connection
    //TODO By the moment we suppose we have enough bandwidth
    std::vector<DataPackage*> data_to_send; //Input and weight temporal storage
    std::vector<DataPackage*> psum_to_send; // psum temporal storage
    this->local_cycle+=1;
    this->sdmemoryStats.total_cycles++; //To track information
    /* CHANGES DONE TO SAVE MEMORY */
    unsigned int current_iteration=current_output_pixel / output_psums_per_channel;
    unsigned index_G=current_G*this->current_tile->get_T_G();
     unsigned index_K=current_K*this->current_tile->get_T_K();
    unsigned int pck_iteration=(index_G)*this->dnn_layer->get_K() + index_K*this->current_tile->get_T_G();
    /* END CHANGES TO SAVE MEMORY */
    if(!this->input_finished && (pck_iteration <= current_iteration)) { //If 
        //1. Weight distribution for the T_K filters
        unsigned window_size = this->current_tile->get_T_R()*this->current_tile->get_T_S()*this->current_tile->get_T_C();
        unsigned folding_shift = 0;
        if(this->current_tile->get_folding_enabled()) { //If there is folding we leave a MS to perform the psum accumulation
           // std::cout << "Aqui hay folding" << std::endl;
            window_size+=1;
            folding_shift+=1;
        }
        if(!this->weights_distributed) {
            //For each weight, decide the receivers
            if(current_tile->get_T_N()*current_tile->get_T_X_()*current_tile->get_T_Y_() > 1) { //If N*X_*Y_ is greater than 1 then MULTICAST MESSAGE as the same weight is sent multiple times.
                for(unsigned g=0; g < current_tile->get_T_G(); g++) {
                    unsigned desp_g = g*current_tile->get_T_K()*current_tile->get_T_X_()*current_tile->get_T_Y_()*window_size;
                    for(unsigned k=0; k < current_tile->get_T_K(); k++) {
                        unsigned desp_k = k*current_tile->get_T_X_()*current_tile->get_T_Y_()*window_size;
                        for(unsigned c=0; c < current_tile->get_T_C(); c++) {
                            for(unsigned r=0; r < current_tile->get_T_R(); r++) {
                                for(unsigned s=0; s < current_tile->get_T_S(); s++) {
                                    //Where do I send this weight
                                    //  To every piece of N
                                    //  To every piece of T_X_
                                    //  To every piece of T_Y_
                                    bool *vector_to_send = new bool[this->num_ms];
                                    //Itiaalizing vector
                                    for(int i=0; i<this->num_ms; i++) {
                                        vector_to_send[i]=false;
                                    }
                                 
                                    for(int n=0; n<this->current_tile->get_T_N(); n++) {
                                        unsigned desp_n = n*this->current_tile->get_T_G()*this->current_tile->get_T_K()*this->current_tile->get_T_X_()*this->current_tile->get_T_Y_()*window_size;
                                        for(int x=0; x<this->current_tile->get_T_X_(); x++) {
                                            for(int y=0; y<this->current_tile->get_T_Y_(); y++) {
                                                unsigned desp_this_neuron = x*this->current_tile->get_T_Y_()*window_size + y*window_size;
                                                vector_to_send[desp_n + desp_g + desp_k + desp_this_neuron +  c*this->current_tile->get_T_S()*this->current_tile->get_T_R() + r*this->current_tile->get_T_S() + s + folding_shift]=true;    //+1 because we skip the first MS again for the folding issue
                                            
                                            }
                                        }
                                    }
                                    //Sending the data
                                    unsigned index_G=current_G*this->current_tile->get_T_G();
                                    unsigned index_K=current_K*this->current_tile->get_T_K();
                                    unsigned index_C=current_C*this->current_tile->get_T_C();
                                    unsigned index_R=current_R*this->current_tile->get_T_R();
                                    unsigned index_S=current_S*this->current_tile->get_T_S();
                                    this->sdmemoryStats.n_SRAM_weight_reads++; //To track information
                                    data_t data = filter_address[(index_G+g)*this->group_size*word_size + (index_K+k)*this->filter_size*word_size + (index_R+r)*this->row_filter_size*word_size + (index_S+s)*dnn_layer->get_C()*word_size + (index_C+c)];  //Fetching. Note the distribution in memory is interleaving the channels
                           
                                    //Creating the package with the weight and the destination vector
                                    DataPackage* pck_to_send = new DataPackage(sizeof(data_t), data, WEIGHT, 0, MULTICAST, vector_to_send, this->num_ms);
                                    //index_K*this->current_tile->get_T_G() because even though index_K iterations have been calculated previously, there are G groups mapped, so really real index_K*T_G
                                    pck_to_send->setIterationK((index_G)*this->dnn_layer->get_K() + index_K*this->current_tile->get_T_G()); //To avoid sending it to the architecture if the output psums of the previous k channels have not been calculated yet.
                                    this->sendPackageToInputFifos(pck_to_send);

                                }
                            }
                        }
                    } //End K
                } //End G
            }
            else { // N*X_*Y_ == 1 so UNICAST MESSAGES. Each weight is sent once
                for(unsigned g=0; g < current_tile->get_T_G(); g++) {
                    for(unsigned k=0; k < current_tile->get_T_K(); k++) {
                        for(unsigned c=0; c < current_tile->get_T_C(); c++) {
                            for(unsigned r=0; r < current_tile->get_T_R(); r++) {
                                for(unsigned s=0; s < current_tile->get_T_S(); s++) {
                                    //Each weight is sent once depending on K (Remember that T_X, T_Y and N is 1 for this tile)
                                    //Create unicast package with the destination of this weight
                                    //Getting the data from memory
                                    unsigned index_G=current_G*this->current_tile->get_T_G();
                                    unsigned index_K=current_K*this->current_tile->get_T_K();
                                    unsigned index_C=current_C*this->current_tile->get_T_C();
                                    unsigned index_R=current_R*this->current_tile->get_T_R();
                                    unsigned index_S=current_S*this->current_tile->get_T_S();
                                    this->sdmemoryStats.n_SRAM_weight_reads++; //To track information
                                    data_t data = filter_address[(index_G+g)*this->group_size*word_size + (index_K+k)*this->filter_size*word_size + 
				        + (index_R+r)*this->row_filter_size*word_size + (index_S+s)*dnn_layer->get_C()*word_size + (index_C+c)];
                                    
			            //Shift of this weight is g*group_tile_size + k*filter_tile_size + c*filter_channel_tile_size + r*s_tile_size + s
			            unsigned int receiver = g*current_tile->get_T_K()*window_size  + k*window_size + 
			    	               c*current_tile->get_T_R()*current_tile->get_T_S() + r*current_tile->get_T_S() + s + folding_shift; //+1 because of the ms we leave free for the folding
			            //Creating the package with the data in memory and the destination
                                    DataPackage* pck_to_send = new DataPackage(sizeof(data_t), data, WEIGHT, 0, UNICAST, receiver);
                                    pck_to_send->setIterationK((index_G)*this->dnn_layer->get_K() + index_K*this->current_tile->get_T_G()); //To avoid sending it to the architecture if the output psums of the previous k channels have not been calculated yet.

                                    this->sendPackageToInputFifos(pck_to_send);
                                }
                            }
                        }
                    } //End K
                } //End G

            }
            this->current_S+=1;
            if(this->current_S == iter_S) { //If all the columns completed
                this->current_R+=1;  //Next tile rows
                this->current_S = 0;
                if(this->current_R == iter_R) { //if all the rows completed
                    this->current_C+=1; //Next tile channels
                    this->current_R = 0;
                    if(this->current_C == iter_C) {
                        this->current_C = 0;
                        //If C is finished then all the weights have been distributed in this point 
                        this->weights_distributed = true;

                    }
                }
            }
        
        } // if(!this->weights_distributed)
        else { //Input distributions. 
            //std::cout << "Sending inputs in the cycle " << this->local_cycle << std::endl;
            unsigned y_inputs = this->current_tile->get_T_S() + (this->current_tile->get_T_Y_()-1)*this->dnn_layer->get_strides(); //Number of input rows per batch of the ifmap obtained from the ofmap dimensions. i.e., filter size + number of extra Y_ Neurons mapped times the extra inputs (stride)
            unsigned x_inputs = this->current_tile->get_T_R() + (this->current_tile->get_T_X_()-1)*(this->dnn_layer->get_strides()); // Nuumber of input cols per batch of the ifmap obtained from the ofmap dimensions
            //unsigned y_inputs = this->current_tile->get_T_Y_() + (this->current_tile->get_T_S()-1); //Number of input rows per batch of the ifmap obtained from the ofmap dimensions
            //unsigned x_inputs = this->current_tile->get_T_X_() + (this->current_tile->get_T_R()-1); // Nuumber of input cols per batch of the ifmap obtained from the ofmap dimensions

            unsigned output_y_inputs = this->current_tile->get_T_Y_() + (this->dnn_layer->get_S()-1);
            unsigned output_x_inputs = this->current_tile->get_T_X_() + (this->dnn_layer->get_R()-1);
            //std::cout << "y_inputs: " << y_inputs << std::endl;
            //std::cout << "x_inputs: " << x_inputs << std::endl;
            //y_init is the first column of the window to send. This might be 0, if all the window must be sent,
            // or the last column if the fw links of the mswitches are enabled in this cycle and therefore bandwidth is saved and just
            //one column must be sent.
            
            unsigned int y_init = 0; //y_init is the first column of the window to send. This might be 0, if all the window must be sent,
            //If the fw links are enabled because the tile allows it                          
            if((this->current_tile->get_T_Y_() == 1) && (this->current_tile->get_T_S()>1) && (this->dnn_layer->get_strides() == 1)) {   //Conditions in which the fw links of the MSwitches are enabled
                if(this->current_Y > 0) { //If it is not the first column of the row, then data is reused among the Mswitches
                    if(this->current_S == (iter_S-1)) { //If it is the last iteration of S
                        y_init = y_inputs - 1;
                    }
                    else { //If it is not the last iteration of the window we do not send anything
                        y_init = y_inputs-1; //Do not send any activation since all the inputs are already in the fifos.
			//CHECK. There was a bug here because y_init was equals to y_inputs. If true, remove the condition
                    }
                }
                else { //If it is the first window interation, we send all the activations in all the iterations of the same window
                    y_init = 0;
                }
            }
           
            for(unsigned x=0; x<x_inputs; x++) {
                for(unsigned y=y_init; y<y_inputs; y++) { //the number of columns to iterate depends on if the fw links of the MS are enabled
                    //Each input is replicated k times for this n
                    //Creating bool for this element                    
                    bool**** destinations = new bool***[this->current_tile->get_T_N()]; //One destination vector for each n and for each c since we are going to send one value (package) per each batch and channel
                    
                    for(int i=0; i<this->current_tile->get_T_N(); i++) { 
                        destinations[i] = new bool**[this->current_tile->get_T_G()]; 
                            for(int g=0; g < this->current_tile->get_T_G(); g++) {
                            destinations[i][g] = new bool*[this->current_tile->get_T_C()];
                            for(int j=0; j<this->current_tile->get_T_C(); j++) {
                                destinations[i][g][j] = new bool[this->num_ms]; //this->num_ms has enough multipliers to cover n
                                for(int z=0;z<this->num_ms;z++) {
                                    destinations[i][g][j][z]=false;
                                }
                            }

                        }
                    }
                    //TODO Me he quedado por aqui y tengo que leiminar una dimension mas en el array destinations por haber metido la dimension g
                   // std::cout << "Buscando VNs para el elemento:  [" << x << "," << y << "]" <<  std::endl;
                    int first_possible_vn_x = (x - (this->current_tile->get_T_R()-1));// / this->dnn_layer->get_strides();
                    if(first_possible_vn_x < 0) {
                        first_possible_vn_x=0;
                    }
                    else {
                        first_possible_vn_x = first_possible_vn_x / this->dnn_layer->get_strides();
                    }
                    int last_possible_vn_x = x / this->dnn_layer->get_strides();
                   // std::cout << "First possible vn x: " << first_possible_vn_x << std::endl;
                   // std::cout << "Last  possible vn y: " << last_possible_vn_x << std::endl;
                    for(int i=first_possible_vn_x; i<=last_possible_vn_x; i++) { //TODO check with strides-1
                        int last_element_vn_i = this->dnn_layer->get_strides()*i + (this->current_tile->get_T_R()-1);
                        int first_element_vn_i = this->dnn_layer->get_strides()*i;
                     //   std::cout << "    Trying VN [" << i << ",-]" << std::endl;
                        if((i>=0) && ((first_element_vn_i+((this->current_tile->get_T_R()-1))) < x_inputs) && (x <= last_element_vn_i) && (x >= first_element_vn_i))  { // If it's a valid vn //This one, not the next
                            // Y loop (cols)
                            int first_possible_vn_y = (y - (this->current_tile->get_T_S()-1));
                            if(first_possible_vn_y < 0 ) {
                                first_possible_vn_y = 0;
                            }

                            else {
                                first_possible_vn_y = first_possible_vn_y / this->dnn_layer->get_strides();
                            }
                            int last_possible_vn_y = y / this->dnn_layer->get_strides();
                            for(int j=first_possible_vn_y; j<=last_possible_vn_y; j++) {
                                int last_element_vn_j = this->dnn_layer->get_strides()*j + (this->current_tile->get_T_S()-1);
                                int first_element_vn_j = this->dnn_layer->get_strides()*j;
                               // std::cout << "        Trying VN [" << i << "," << j << "]" << std::endl;
                                if((j >= 0) && ((first_element_vn_j+(this->current_tile->get_T_S()-1)) < y_inputs) && (y <= last_element_vn_j) && (y >= first_element_vn_j)) { //If it's a valid vn
                                    //Enable this for every k and n
                                    int current_vn = i*this->current_tile->get_T_Y_() + j; //i times the row size + desp of the currrent row
                                 //   std::cout << "Current VN SELECTED: " << current_vn << std::endl;
                                  //  std::cout << "            i: " << i << std::endl;
                                   // std::cout << "            j: " << j << std::endl;
                                    int desp_y_inside_vn = y - j*this->dnn_layer->get_strides(); //Shift y (cols) inside vn
                                    int desp_x_inside_vn = x - i*this->dnn_layer->get_strides(); //Shift x (rows) inside vn
                                    
                                   // std::cout << "            Desp_x_inside_vn: " << desp_x_inside_vn << std::endl;
                                    //std::cout << "            Desp_y_inside_vn: " << desp_y_inside_vn << std::endl;
                                    int desp_total_inside_vn = desp_x_inside_vn*this->current_tile->get_T_S() + desp_y_inside_vn;
                                       
                                    //std::cout << "Desp_total_inside_vn: " << desp_total_inside_vn << std::endl;
                                    int current_receiver = current_vn*this->current_tile->get_VN_Size() + desp_total_inside_vn;    //Receiver for an arbitry n and k. //+1 for the folding
                                    if(this->current_tile->get_folding_enabled()) {
                                        current_receiver+=1; //1 ms free for psum accumulation
                                    }
                                    for(int n=0; n<this->current_tile->get_T_N(); n++) { //For every n (batch size)  
                             
                                        int desp_n = n*this->current_tile->get_T_K()*this->current_tile->get_T_X_()*this->current_tile->get_T_Y_()*window_size; //Desp n = n*t_k*t_x_*t_y_
                                        for(int g=0; g<this->current_tile->get_T_G(); g++) {
                                            int desp_g = g*this->current_tile->get_T_K()*this->current_tile->get_T_X_()*this->current_tile->get_T_Y_()*window_size;
                                            for(int k=0; k<this->current_tile->get_T_K(); k++) {
                                                int desp_k = k*this->current_tile->get_T_X_()*this->current_tile->get_T_Y_()*window_size;
                                                //std::cout << "desp_k: " << desp_k << std::endl;
                                                for(int c=0; c<this->current_tile->get_T_C(); c++) { //For each channel //TODO the mistake is here
                                                    int desp_c = c*this->current_tile->get_T_R()*this->current_tile->get_T_S(); //jump channel size
                                                    destinations[n][g][c][desp_n + desp_g +  desp_k+ desp_c + current_receiver]=true; //The package for n will have the desp_n+desp_k ms enabled to receive the activation
                                                }
                                            
                                            }  //End K
                                        } //End G
                                    } //End N
                                }
                            }
                        }
                    }

                    //Taking the data for each n and send using the corresponding bit_vector in several packages
                     unsigned index_N=current_N*this->current_tile->get_T_N();
                     unsigned index_G=current_G*this->current_tile->get_T_G();
                     unsigned index_K=current_K*this->current_tile->get_T_K();
                     unsigned index_X=current_X*this->current_tile->get_T_X_();
                     unsigned index_Y=current_Y*this->current_tile->get_T_Y_();
                     unsigned index_C=current_C*this->current_tile->get_T_C();
                     unsigned index_R=current_R*this->current_tile->get_T_R();
                     unsigned index_S=current_S*this->current_tile->get_T_S();
                     

                    for(int i=0; i<this->current_tile->get_T_N(); i++) {
                        for(int g=0; g<this->current_tile->get_T_G(); g++) {
                            for(int c=0; c<this->current_tile->get_T_C(); c++) {
                                bool* destination_vector = destinations[i][g][c];
                               // for(int i=0; i<this->num_ms; i++) 
                                //    std::cout << destination_vector[i];
                                //std::cout << std::endl;
                                this->sdmemoryStats.n_SRAM_input_reads++; 
                                data_t data = input_address[(index_N+i)*this->input_size+((index_X*this->dnn_layer->get_strides()+ x) + index_R)*this->dnn_layer->get_Y()*this->dnn_layer->get_C()*this->dnn_layer->get_G()*word_size+((index_Y*this->dnn_layer->get_strides() + y) + index_S)*this->dnn_layer->get_C()*this->dnn_layer->get_G()*word_size + (index_G+g)*dnn_layer->get_C()*word_size + (index_C+c)*word_size]; //Read value input(x,y)
                                //Creating multicast package. Even though the package was unicast, multicast format is used anyway with just one element true in the destination vector
                                DataPackage* pck = new DataPackage(sizeof(data_t), data,IACTIVATION,0, MULTICAST, destination_vector, this->num_ms);
                                pck->setIterationK((index_G)*this->dnn_layer->get_K() + index_K*this->current_tile->get_T_G()); //To avoid sending it to the architecture if the output psums of the previous k channels have not been calculated yet.

                                this->sendPackageToInputFifos(pck);
                                //destinations[i][g][c] is not deleted as it is used in the package
                            } //End C
                            delete[] destinations[i][g];
                        } //End G
                            delete[] destinations[i]; //deleting array with pointers to bool* but not the bool* itself
                    } //End i TODO REMOVE other dimension for G
                    delete[] destinations; //deelting array with pointers to bool**
                } 
            }

            //TODO iter_X deberia de ser iter_X de inputs, no?         
            //Updating variables
            this->current_S+=1;
            if(this->current_S == this->iter_S) {
                this->current_S = 0;
                this->current_R+=1;
                if(this->current_R == iter_R) {
                    this->current_R = 0;
                    this->current_C+=1;
                    if(this->current_C == iter_C) {
                        this->current_C = 0;
                        this->current_Y+=1; //Updating rows
                            if(this->current_Y==this->iter_Y) {
                            //Updating X
                                this->current_Y=0; //Reset Y
                                this->current_X+=1;
                                if(this->current_X == this->iter_X) {
                                    //If rows finished, updating next N batch
                                    this->current_X = 0; //Updating X
                                    this->current_K+=1;
                                    this->weights_distributed = false; //Distribution enabled for next iteration
                                    if(this->current_K==iter_K) {
                                        this->current_K=0;
                                        this->current_G+=1;
                                        if(this->current_G==this->iter_G) {
                                            this->current_G=0;
                                            this->current_N+=1;
                                            if(this->current_N==this->iter_N) {
                                            //Update K. Weight distribution neccesary first!
                                                this->current_N = 0;
                                                this->weights_distributed = true; //Avoid distribution
                                                this->weights_finished=true;
                                                //Current_K updated when weights are distributed
                                                //Checking if all the inputs have been delivered ()
                                                if(weights_finished) {
                                                    this->input_finished=true;
                                                }
                                            } //end iter_n
                                        } //end iter_g
                                    }
                                } //end iter_x

                            } //end iter_y


                    } //end iter_C 
                } //end iter_S
            } //end iter_R
            
        
        }
    //Sending the vector
    //std::cout << "Size vector to send: " << data_to_send.size() << std::endl;
    //for(int i=0; i<data_to_send.size(); i++) {
    //    std::cout << "Data package: " << data_to_send[i]->get_data() << " Type: " << data_to_send[i]->get_data_type() << std::endl;
        
   // }
    
    } //End if input_finished

    
       
 
         

    //Receiving output data from write_connection
    this->receive();
    if(!write_fifo->isEmpty()) {
        //Index the data by using the VN Address Table and the VN id of the packages
        for(int i=0; i<write_fifo->size(); i++) {
            DataPackage* pck_received = write_fifo->pop();
            unsigned int vn = pck_received->get_vn();
            data_t data = pck_received->get_data();
            //std::cout << "Writing data: " << data << std::endl;
            // using the VNAT register to get the address to write
            assert(vn==VNAT[vn]->VN);
            //std::cout << "Memory received a psum " << data << std::endl;
            unsigned int addr_offset = this->VNAT[vn]->addr;
            this->sdmemoryStats.n_SRAM_psum_writes++; //To track information 
            this->output_address[addr_offset]=data; //ofmap or psum, it does not matter.
            //std::cout << "value written " << data << std::endl;
            current_output_pixel+=1; 
            this->sdmemoryStats.n_SRAM_write_ports_use[pck_received->getOutputPort()]++; //To track information
#ifdef DEBUG_MEM_OUTPUT
            std::cout << "[MEM_OUTPUT] Cycle " <<  local_cycle  << ", STONNE generated a partial output from output port " <<  pck_received->getOutputPort() << " and VN " << pck_received->get_vn() << ". OutputCount ( " << current_output_pixel << "/" << this->output_pixels_to_compute << ")" << std::endl;
#endif
           if((current_output_pixel % 10000) == 0) {
            std::cout << "[MEM_OUTPUT] Cycle " <<  local_cycle  << ", STONNE generated a partial output from output port " <<  pck_received->getOutputPort() << " and VN " << pck_received->get_vn() << ". OutputCount ( " << current_output_pixel << "/" << this->output_pixels_to_compute << ")" << std::endl;
           }

           //std::cout << "Writting in position " << addr_offset << " the value of " << pck_received->get_data() <<  std::endl;

             
            this->VNAT[vn]->current_psum+=1;
            //If forwarding is disabled (current_tile->folding_enabled=false) then this->VNAT[vn]->n_psums will be 1 and therefore, the next condition will neve be true and the psum will never be sent.
            if(this->VNAT[vn]->current_psum < this->VNAT[vn]->n_psums) { //Forward psum data to perform the next psum //
                bool* destination_vector = new bool[this->num_ms];
                for(int i=0; i<this->num_ms; i++) {
                    destination_vector[i]=false;
                }
                destination_vector[vn*this->current_tile->get_VN_Size()]=true;
                DataPackage* pck = new DataPackage(sizeof(data_t), data, PSUM,0, MULTICAST, destination_vector, this->num_ms);
                this->sdmemoryStats.n_SRAM_psum_reads++; //To track information
                this->sendPackageToInputFifos(pck); //Sending the package to fifos
            }
            else {
                this->VNAT[vn]->current_psum=0; //Updating 
            }
           // std::cout << "OUTPUT_PIXELS_TO_COMPUTE: " << output_pixels_to_compute << std::endl;
            //std::cout << "CURRENT_OUTPUT_PIXEL: " << current_output_pixel << std::endl;
            if(current_output_pixel == output_pixels_to_compute) {
                this->execution_finished = true;
            }
            this->VNAT[vn]->update(); //Calculate next output address for that vn
            //Updating address
             
            //std::cout << "Package " << pck_received->get_data() << " received with vn: ";
            //std::cout << vn << std::endl;
           // address_t addr = output_address +
           delete pck_received; //Deleting the current package
            
        }
    }

    //Introducing temporal data to send to the FIFOs used to send 


    this->send(); //Send the content in read_fifo according to the current bw
}

bool SDMemory::isExecutionFinished() {
    return this->execution_finished;
}

/* The traffic generation algorithm generates a package that contains a destination for all the ms. We have to divide it into smaller groups of ms since they are divided into several ports */
void SDMemory::sendPackageToInputFifos(DataPackage* pck) {
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

void SDMemory::send() {
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
           
            //calculating the current k iteration. i.e., n psums already calculated / number of psums per each channel iteration
            unsigned int current_iteration=current_output_pixel / output_psums_per_channel;
            /*std::cout << "current_output_pixel: " << current_output_pixel << std::endl;
            std::cout << "output_psums_per_channel: " << output_psums_per_channel << std::endl;
            std::cout << "Current_iteration: " << current_iteration << std::endl;
            std::cout << "pck iteration: " << pck->getIterationK() << std::endl; */
            if(pck->getIterationK() <= current_iteration) { //Check if the iteration of the package is older or equal to the current
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
            else {
#ifdef DEBUG_MEM_INPUT
    std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", input port " << i << " waiting for iteration " << current_iteration << std::endl;
#endif
            }            

        }
            
    }
}

//TODO Remove this connection
void SDMemory::receive() { //TODO control if there is no space in queue
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

void SDMemory::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"SDMemoryStats\" : {" << std::endl; //TODO put ID
    this->sdmemoryStats.print(out, indent+IND_SIZE);
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void SDMemory::printEnergy(std::ofstream& out, unsigned int indent) {
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

