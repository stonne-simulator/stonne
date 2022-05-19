//
// Created by Francisco Munoz-Martinez on 18/06/19.
//
#include "STONNEModel.h"

#include <assert.h>
#include <chrono>
#include "types.h"
#include <vector>
#include "Tile.h"
#include "utility.h"
#include "Config.h"
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "cpptoml.h"

Stonne::Stonne(Config stonne_cfg) {
    this->stonne_cfg=stonne_cfg;
    this->ms_size = stonne_cfg.m_MSNetworkCfg.ms_size;
    this->layer_loaded=false;
    this->tile_loaded=false;
    this->outputASConnection = new Connection(stonne_cfg.m_SDMemoryCfg.port_width);
    this->outputLTConnection = new Connection(stonne_cfg.m_LookUpTableCfg.port_width);
    switch(stonne_cfg.m_MSNetworkCfg.multiplier_network_type) {
        case LINEAR: 
	    this->msnet = new MSNetwork(2, "MSNetwork", stonne_cfg);
	    break;
	case OS_MESH:
	    this->msnet = new OSMeshMN(2, "OSMesh", stonne_cfg);
	    break;
	default:
	    assert(false);
    }
    //switch(DistributionNetwork). It is possible to create instances of other DistributionNetworks.h
    this->dsnet = new DSNetworkTop(1, "DSNetworkTop", stonne_cfg);
    
    //Creating the ReduceNetwork according to the parameter specified by the user
    switch(stonne_cfg.m_ASNetworkCfg.reduce_network_type) {
    case ASNETWORK:
        this->asnet = new ASNetwork(3, "ASNetwork", stonne_cfg, outputASConnection); 
        break;
    case FENETWORK:
        this->asnet = new FENetwork(3, "FENetwork", stonne_cfg, outputASConnection);
        break;
    case TEMPORALRN:
	this->asnet = new TemporalRN(3, "TemporalRN", stonne_cfg, outputASConnection);
	break;
    default:
	assert(false);
    }

    this->collectionBus = new Bus(4, "CollectionBus", stonne_cfg); 
    this->lt = new LookupTable(5, "LookUpTable", stonne_cfg, outputASConnection, outputLTConnection);

    //switch(MemoryController). It is possible to create instances of other MemoryControllers
    switch(stonne_cfg.m_SDMemoryCfg.mem_controller_type) {
	case SIGMA_SPARSE_GEMM:
            this->mem = new SparseSDMemory(0, "SparseSDMemory", stonne_cfg, this->outputLTConnection);
	    break;
	case MAERI_DENSE_WORKLOAD:
	    this->mem = new  SDMemory(0, "SDMemory", stonne_cfg, this->outputLTConnection);
	    break;
	case MAGMA_SPARSE_DENSE:
            this->mem = new  SparseDenseSDMemory(0, "SparseDenseSDMemory", stonne_cfg, this->outputLTConnection);
            break;
	case TPU_OS_DENSE:
	    this->mem = new  OSMeshSDMemory(0, "OSMeshSDMemory", stonne_cfg, this->outputLTConnection);
	    break;
	default:
	    assert(false);
    }
    //Adding to the memory controller the asnet and msnet to reconfigure them if needed
    this->mem->setReduceNetwork(asnet);
    this->mem->setMultiplierNetwork(msnet); 

    //Calculating n_adders
    this->n_adders=this->ms_size-1; 
    //rsnet
    this->connectMemoryandDSN();
    this->connectMSNandDSN();
    this->connectMSNandASN();

    this->connectASNandBus();
    this->connectBusandMemory();
  
    //DEBUG PARAMETERS
    this->time_ds = 0;
    this->time_ms = 0;
    this->time_as = 0;
    this->time_lt = 0;
    this->time_mem = 0;

    //STATISTICS
    this->n_cycles = 0;

}

Stonne::~Stonne() {
    delete this->dsnet;
    delete this->msnet;
    delete this->asnet;
    delete this->outputASConnection;
    delete this->outputLTConnection;
    delete this->lt;
    delete this->mem;
    delete this->collectionBus;
    if(layer_loaded) {
        delete this->dnn_layer;
    }
  
    if(tile_loaded) {
        delete this->current_tile;
    } 
}

//Connecting the DSNetworkTop input ports with the read ports of the memory. These connections have been created
//by the module DSNetworkTop, so we just have to connect them with the memory.
void Stonne::connectMemoryandDSN() {
    std::vector<Connection*> DSconnections = this->dsnet->getTopConnections();
    //Connecting with the memory
    this->mem->setReadConnections(DSconnections);
}

//Connecting the multipliers of the mSN to the last level switches of the DSN. In order to do this link correct, the number of 
//connections in the last level of the DSN (output connections of the last level switches) must match the number of multipliers. 
//The multipliers are then connected to those connections, setting a link between them. 
void Stonne::connectMSNandDSN() {
    std::map<int, Connection*> DNConnections = this->dsnet->getLastLevelConnections(); //Map with the DS connections
    this->msnet->setInputConnections(DNConnections);
     
}
//Connect the multiplier switches with the Adder switches. Note the number of ASs connection connectionss and MSs must be the identical

void Stonne::connectMSNandASN() {
    std::map<int, Connection*> RNConnections = this->asnet->getLastLevelConnections(); //Map with the AS connections
    this->msnet->setOutputConnections(RNConnections);

}

void Stonne::connectASNandBus() {
        std::vector<std::vector<Connection*>> connectionsBus = this->collectionBus->getInputConnections(); //Getting the CollectionBus Connections
        this->asnet->setMemoryConnections(connectionsBus); //Send the connections to the ReduceNetwork to be connected according to its algorithm
   
   
    
}

void Stonne::connectBusandMemory() {
    std::vector<Connection*> write_port_connections = this->collectionBus->getOutputConnections();
    this->mem->setWriteConnections(write_port_connections);
       
}

void Stonne::loadDNNLayer(Layer_t layer_type, std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G, unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, address_t input_address, address_t filter_address, address_t output_address, Dataflow dataflow) {
    assert((C % G)==0); //G must be multiple of C
    assert((K % G)==0); //G must be multiple of K
    assert(X>=R);
    assert(Y>=S);
    if((layer_type==FC) || (layer_type==GEMM)) {
        //assert((R==1) && (C==1) && (G==1) && (Y==S) && (X==1)); //Ensure the mapping is correct
    } 
    this->dnn_layer = new DNNLayer(layer_type, layer_name, R,S, C, K, G, N, X, Y, strides);   
    this->layer_loaded = true;
    this->mem->setLayer(this->dnn_layer, input_address, filter_address, output_address, dataflow);
}

void Stonne::loadCONVLayer(std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G, unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, address_t input_address, address_t filter_address, address_t output_address) {
    loadDNNLayer(CONV, layer_name, R, S, C, K, G, N, X, Y, strides, input_address, filter_address, output_address, CNN_DATAFLOW);
    std::cout << "Loading a convolutional layer into STONNE" << std::endl;
}

void Stonne::loadFCLayer(std::string layer_name, unsigned int N, unsigned int S, unsigned int K, address_t input_address, address_t filter_address, address_t output_address)  {
     //loadDNNLayer(FC, layer_name, 1, S, 1, K, 1, N, 1, S, 1, input_address, filter_address, output_address, CNN_DATAFLOW);
    loadDNNLayer(FC, layer_name, 1, S, 1, K, 1, 1, N, S, 1, input_address, filter_address, output_address, CNN_DATAFLOW);
    std::cout << "Loading a FC layer into STONNE" << std::endl;
}

void Stonne::loadGEMM(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, metadata_address_t MK_metadata, metadata_address_t KN_metadata, address_t output_matrix, metadata_address_t output_metadata, Dataflow dataflow) {
    //Setting GEMM (from SIGMA) parameters onto CNN parameters:
    //N=N
    //S and X in CNN =K in SIGMA
    //K in CNN = M in SIGMA
    //input_matrix=KN 
    //filter_matrix = MK
    loadDNNLayer(GEMM, layer_name, 1, K, 1, M, 1, 1, N, K, 1, MK_matrix, KN_matrix, output_matrix, dataflow);
    std::cout << "Loading a GEMM into STONNE" << std::endl;
    this->mem->setSparseMetadata(MK_metadata, KN_metadata, output_metadata); 
    std::cout << "Loading metadata" << std::endl;
}

void Stonne::loadDenseGEMM(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, address_t output_matrix, Dataflow dataflow) {
    //Setting GEMM (from SIGMA) parameters onto CNN parameters:
    //N=N
    //S and X in CNN =K in SIGMA
    //K in CNN = M in SIGMA
    //input_matrix=KN
    //filter_matrix = MK
    loadDNNLayer(GEMM, layer_name, 1, K, 1, N, 1, 1, M, K, 1, MK_matrix, KN_matrix, output_matrix, dataflow);
    std::cout << "Loading a GEMM into STONNE" << std::endl;
}

void Stonne::loadSparseDense(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, address_t output_matrix, unsigned int T_N, unsigned int T_K) {
    //Setting GEMM (from SIGMA) parameters onto CNN parameters:
    //K in CNN=N here
    //C in CNN =K here
    //N in CNN = M here
    //input_matrix=MK 
    //filter_matrix = KN
    loadDNNLayer(SPARSE_DENSE, layer_name, 1, 1, K, N, 1, M, 1, 1, 1, MK_matrix, KN_matrix, output_matrix, SPARSE_DENSE_DATAFLOW);
    std::cout << "Loading a Sparse multiplied by dense GEMM into STONNE" << std::endl;
    /////To define in the new class
    this->mem->setSparseMatrixMetadata(MK_metadata_id, MK_metadata_pointer);
    std::cout << "Loading metadata" << std::endl;

    /////To define in the new class
    loadSparseDenseTile(T_N, T_K);
}

//To dense CNNs and GEMMs 
void Stonne::loadTile(unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G, unsigned int T_N, unsigned int T_X_, unsigned int T_Y_) {

    assert(this->layer_loaded);
    if(stonne_cfg.m_MSNetworkCfg.multiplier_network_type==LINEAR) {
        assert(this->ms_size >= (T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_)); //There are enough mswitches
    }
    else {
        assert((this->stonne_cfg.m_MSNetworkCfg.ms_rows*this->stonne_cfg.m_MSNetworkCfg.ms_cols) >= (T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_));
    }
    //Checking if the dimensions fit the DNN layer. i.e., the tile is able to calculate the whole layer.
    std::cout << "Loading Tile: <T_R=" << T_R << ", T_S=" << T_S << ", T_C=" << T_C << ", T_K=" << T_K << ", T_G=" << T_G << ", T_N=" << T_N << ", T_X'=" << T_X_ << ", T_Y'=" << T_Y_ << ">" << std::endl; 
 
    //Remove these lines if we want the architeture to compute the layer even if the tile does not fit. 
    // This will mean that some row, columns or output channels would remain without calculating. 
    if(stonne_cfg.m_SDMemoryCfg.mem_controller_type==MAERI_DENSE_WORKLOAD) { //Just for this maeri controller
       // assert((this->dnn_layer->get_R() % T_R) == 0);    // T_R must be multiple of R
       // assert((this->dnn_layer->get_S() % T_S) == 0);    // T_S must be multiple of S
       // assert((this->dnn_layer->get_C() % T_C) == 0);    // T_C must be multiple of C
       // assert((this->dnn_layer->get_K() % T_K) == 0);    // T_K must be multiple of K
       // assert((this->dnn_layer->get_G() % T_G) == 0);    // T_G must be multiple of G
       // assert((this->dnn_layer->get_N() % T_N) == 0);    // T_N must be multiple of N
       // assert((this->dnn_layer->get_X_() % T_X_) == 0);  // T_X_ must be multiple of X_
       // assert((this->dnn_layer->get_Y_() % T_Y_) == 0);  // T_Y_ must be multiple of Y_ 
    }

    //End check
    unsigned int n_folding = ceil(this->dnn_layer->get_R() / (float) T_R)*ceil(this->dnn_layer->get_S() / (float)T_S) * ceil(this->dnn_layer->get_C() / (float)T_C) ;
    bool folding_enabled = false; //Condition to use extra multiplier. Note that if folding is enabled but some type of accumulation buffer is needed this is false as no fw ms is needed. 
    if((n_folding > 1) && (this->stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled==0) && (this->stonne_cfg.m_ASNetworkCfg.reduce_network_type != FENETWORK)) { //If there is folding and the RN is not able to acumulate itself, we have to use an extra MS to accumulate
        folding_enabled = true; 
        //When there is folding we leave one MS free per VN aiming at suming the psums. In next line we check if there are
        // enough mswitches in the array to support the folding. 
        assert(this->ms_size >= ((T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_) + (T_K*T_G*T_N*T_X_*T_Y_))); //We sum one mswitch per VN
    }
    this->current_tile = new Tile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_, folding_enabled);
    
    //Generating the signals for the reduceNetwork and configuring it. The asnet->configureSignals will call its corresponding compiler to generate the signals and allocate all of them
    if(this->stonne_cfg.m_MSNetworkCfg.multiplier_network_type != OS_MESH) { //IN TPU the configuration is done in the mem controller
        this->asnet->configureSignals(this->current_tile, this->dnn_layer, this->ms_size, n_folding); //Calling the ART to configure the signals with them previously generated 
    //Getting MN signals
        this->msnet->configureSignals(this->current_tile, this->dnn_layer, this->ms_size, n_folding);
    }
    //Setting the signals to the corresponding networks

    //If stride > 1 then all the signals of ms_fwreceive_enabled and ms_fwsend_enabled must be disabled since no reuse between MSwitches can be done. In order to not to incorporate stride
    //as a tile parameter, we leave the class Tile not aware of the stride. Then, if stride exists, here the possible enabled signals (since tile does not know about tile) are disabled.
    this->tile_loaded = true;
    this->mem->setTile(this->current_tile);
}

void Stonne::loadGEMMTile(unsigned int T_N, unsigned int T_K, unsigned int T_M)  {
    //loadTile(1, T_K, 1, T_M, 1, T_N, 1, 1);
    std::cout << "Loading a GEMM tile" << std::endl;
    loadTile(1, T_K, 1, T_N, 1, 1, T_M, 1);
    assert(this->layer_loaded && (this->dnn_layer->get_layer_type() == GEMM));   //Force to have the right layer with the GEMM parameters)
}

void Stonne::loadFCTile(unsigned int T_S, unsigned int T_N, unsigned int T_K)  {
    //loadTile(1, T_S, 1, T_K, 1, T_N, 1, 1);
    std::cout << "Loading a FC tile" << std::endl;
    loadTile(1, T_S, 1, T_K, 1, 1, T_N, 1);
    assert(this->layer_loaded && (this->dnn_layer->get_layer_type() == FC));   //Force to have the right layer with the FC parameters)
}

void Stonne::loadSparseDenseTile(unsigned int T_N, unsigned int T_K) {
    std::cout << "Loading SparseDense tile: <T_N=" << T_N << ", T_K=" << T_K << ">" << std::endl;
    this->mem->setDenseSpatialData(T_N, T_K);
}


void Stonne::loadTile(std::string tile_file) {
    auto config = cpptoml::parse_file(tile_file); //Creating object to parse
    auto tile_type=config->get_as<std::string>("tile_type");
    auto T_R=config->get_as<int32_t>("T_R");
    auto T_S=config->get_as<int32_t>("T_S");
    auto T_C=config->get_as<int32_t>("T_C");
    auto T_K=config->get_as<int32_t>("T_K");
    auto T_G=config->get_as<int32_t>("T_G");
    auto T_N=config->get_as<int32_t>("T_N");
    auto T_X_=config->get_as<int32_t>("T_X'");
    auto T_Y_=config->get_as<int32_t>("T_Y'");
    auto tileGeneratorTarget_str=config->get_as<std::string>("generate_tile");
    auto tileGenerator_str=config->get_as<std::string>("generator");
    TileGenerator::Target tileGeneratorTarget = TileGenerator::Target::NONE;
    TileGenerator::Generator tileGenerator = TileGenerator::Generator::CHOOSE_AUTOMATICALLY;

    if(!tile_type) {
        std::cout << "Error to parse tile_type. Parameter not found" << std::endl;
        exit(1);
    }

    if(*tile_type=="CONV") { //Actually the architecture does not know about the layer type. This is just to make sure that the user introduces the
        //appropiate parameters.
        std::cout << "Reading a tile of type CONV" << std::endl;
    }

    else if(*tile_type=="FC") {
        std::cout << "Reading a tile of type FC" << std::endl;
    }

    else {
        std::cout << "Error to parse tile_type. Specify a correct type: [CONV, FC, POOL]" << std::endl;
        exit(1);
    }

    if(*tile_type=="CONV") {
        // if generateTile is specified then generate a FC tile configuration for the layer
        if (tileGeneratorTarget_str && (tileGeneratorTarget = parseTileGeneratorTarget(*tileGeneratorTarget_str)) != TileGenerator::Target::NONE) {
            if (tileGenerator_str)
                tileGenerator = parseTileGenerator(*tileGenerator_str);
            generateTile(tileGenerator, tileGeneratorTarget);
            return;
        }
        // in other case, parse all arguments and load the tile
        else {
            if(!T_R) {
                std::cout << "Error to parse T_R. Value not found." << std::endl;
                exit(1);
            }

            if(!T_S) {
                std::cout << "Error to parse T_S. Value not found." << std::endl;
                exit(1);
            }

            if(!T_C) {
                std::cout << "Error to parse T_C. Value not found." << std::endl;
                exit(1);
            }

            if(!T_K) {
                std::cout << "Error to parse T_K. Value not found." << std::endl;
                exit(1);
            }

            if(!T_G) {
                std::cout << "Error to parse T_G. Value not found." << std::endl;
                exit(1);
            }

            if(!T_N) {
                std::cout << "Error to parse T_N. Value not found." << std::endl;
                exit(1);
            }

            if(!T_X_) {
                std::cout << "Error to parse T_X'. Value not found." << std::endl;
                exit(1);
            }

            if(!T_Y_) {
                std::cout << "Error to parse T_Y'. Value not found." << std::endl;
                exit(1);
            }

            //Filling the parameters
            loadTile(*T_R, *T_S, *T_C, *T_K, *T_G, *T_N, *T_X_, *T_Y_);
        }

    }

    else if(*tile_type=="FC") {
        // if generateTile is specified then generate a FC tile configuration for the layer
        if (tileGeneratorTarget_str && (tileGeneratorTarget = parseTileGeneratorTarget(*tileGeneratorTarget_str)) != TileGenerator::Target::NONE) {
            if (tileGenerator_str)
                tileGenerator = parseTileGenerator(*tileGenerator_str);
            generateTile(tileGenerator, tileGeneratorTarget);
            return;
        }
        // in other case, parse all arguments and load the tile
        else {
            if(!T_N) {
                std::cout << "Error to parse T_N. Value not found." << std::endl;
                exit(1);
            }

            if(!T_S) {
                std::cout << "Error to parse T_S. Value not found." << std::endl;
                exit(1);
            }

            if(!T_K) {
                std::cout << "Error to parse T_K. Value not found." << std::endl;
                exit(1);
            }

            //Filling the parameters
            loadFCTile(*T_S, *T_N, *T_K);
        }
    }

    //Folding is not specified in this case since this use case is not to load the tile into the architecture. Rather, it is to load the tile from the file and layer specify all the parameters
    // to the architecture by means of some abstractions like an instruction.

} //End parser

// General tile generation function for each type of layer
// The sparsity (if specified) must be between 0 and 1
void Stonne::generateTile(TileGenerator::Generator generator, TileGenerator::Target target, float MK_sparsity) {
    std::cout << "Generating a tile automatically..." << std::endl;
    std::cout << "Using generator <" << parseTileGenerator(generator) << "> and target <" << parseTileGeneratorTarget(target) << ">" << std::endl;

    assert(MK_sparsity >= 0 && MK_sparsity <= 1); // implementation check, user will not see it

    TileGenerator::TileGenerator tileGenerator(stonne_cfg.m_MSNetworkCfg.ms_size,
                                               stonne_cfg.m_SDMemoryCfg.n_read_ports,
                                               stonne_cfg.m_SDMemoryCfg.n_write_ports,
                                               generator);

    switch (dnn_layer->get_layer_type()) {
        case Layer_t::CONV: { // Generates a tile using the Stonne CONV parameters
            unsigned int R = dnn_layer->get_R();
            unsigned int S = dnn_layer->get_S();
            unsigned int C = dnn_layer->get_C();
            unsigned int K = dnn_layer->get_K();
            unsigned int G = dnn_layer->get_G();
            unsigned int N = dnn_layer->get_N();
            unsigned int X = dnn_layer->get_X();
            unsigned int Y = dnn_layer->get_Y();
            unsigned int strides = dnn_layer->get_strides();
            unsigned int X_ = (X - R + strides) / strides;
            unsigned int Y_ = (Y - S + strides) / strides;

            // Generate tile and print it on the screen
            TileGenerator::ConvTile tile = tileGenerator.generateConvTile(R, S, C, K, G, N, X, Y, X_, Y_, strides, target);

            std::cout << "Generated tile: <T_R=" << tile.T_R << ", T_S=" << tile.T_S << ", T_C=" << tile.T_C << ", T_K=" << tile.T_K << ", T_G=" << tile.T_G << ", T_N=" << tile.T_N << ", T_X'=" << tile.T_X_ << ", T_Y'=" << tile.T_Y_ << ">" << std::endl;

            // Loads the generated tile and checks parameters
            loadTile(tile.T_R, tile.T_S, tile.T_C, tile.T_K, tile.T_G, tile.T_N, tile.T_X_, tile.T_Y_);
            break;
        }

        case Layer_t::FC:
        case Layer_t::GEMM: { // Generates a tile using the Stonne FC parameters
            // See Stonne::loadDenseGEMM for reference to this map
            unsigned int M = dnn_layer->get_X();
            unsigned int N = dnn_layer->get_K();
            unsigned int K = dnn_layer->get_S();

            // Generate tile and get it's fields
            TileGenerator::DenseGemmTile tile = tileGenerator.generateDenseGemmTile(M, N, K, target);

            std::cout << "Generated tile: <T_M=" << tile.T_M << ", T_N=" << tile.T_N << ", T_K=" << tile.T_K << ">" << std::endl;

            // Loads the generated tile and checks parameters
            if (dnn_layer->get_layer_type() == Layer_t::FC)
                loadFCTile(tile.T_K, tile.T_M, tile.T_N);
            else // dnn_layer->get_layer_type() == Layer_t::GEMM
                loadGEMMTile(tile.T_N, tile.T_K, tile.T_M);

            break;
        }

        case Layer_t::SPARSE_DENSE: { // Generates a tile using the Stonne SparseDense parameters
            unsigned long long M = dnn_layer->get_N();
            unsigned long long N = dnn_layer->get_K();
            unsigned long long K = dnn_layer->get_C();

            // Generate tile and get it's fields
            TileGenerator::SparseDenseTile tile = tileGenerator.generateSparseDenseTile(M, N, K, MK_sparsity, target);

            std::cout << "Generated tile: <T_N=" << tile.T_N << ", T_K=" << tile.T_K << ">" << std::endl;


            // Loads the generated tile and checks parameters
            loadSparseDenseTile(tile.T_N, tile.T_K);
            break;
        }
        default: {
            std::cout << "Error: Unknown layer type." << std::endl;
            assert(false);
        }
    }

}


void Stonne::run() {
    //Execute the cycles
    this->cycle();
}


void Stonne::cycle() {
    //this->testDSNetwork(this->ms_size);
    //this->testTile(this->ms_size);
    //this->printStats();
    bool execution_finished=false;
    while(!execution_finished) {
        auto start = std::chrono::steady_clock::now();
        this->mem->cycle();
        auto end = std::chrono::steady_clock::now();
        this->time_mem+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        start = std::chrono::steady_clock::now();
        //this->lt->cycle();
        end = std::chrono::steady_clock::now();
        this->time_lt+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        this->collectionBus->cycle(); 
        start = std::chrono::steady_clock::now();
        this->asnet->cycle();
        this->lt->cycle();
//        this->collectionBus->cycle(); //This order since these are connections that have to be seen in next cycle
        end = std::chrono::steady_clock::now();
        this->time_as+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        start = std::chrono::steady_clock::now();
        this->msnet->cycle();
        end = std::chrono::steady_clock::now();
        this->time_ms+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        start = std::chrono::steady_clock::now();
        this->dsnet->cycle();
        end = std::chrono::steady_clock::now();
        this->time_ds+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        execution_finished = this->mem->isExecutionFinished();
        this->n_cycles++;
    }

    if(this->stonne_cfg.print_stats_enabled) { //If sats printing is enable
        this->printStats();
        this->printEnergy();
    }
    std::cout << "Number of cycles running: " << this->n_cycles << std::endl;
    std::cout << "Time mem: " << time_mem/1000000 << std::endl;
    std::cout << "Time lt: " << time_lt/1000000 << std::endl;
    std::cout << "Time as: " << time_as/1000000 << std::endl;
    std::cout << "Time ms: " << time_ms/1000000 << std::endl;
    std::cout << "Time ds: " << time_ds/1000000 << std::endl;
    //std::cout << "Time routing ds: " << this->dsnet->get_time_routing()/1000000000 << std::endl;
}

//General function to print all the STATS
void Stonne::printStats() {
    std::cout << "Printing stats" << std::endl;

    std::ofstream out; 
    unsigned int num_ms = this->stonne_cfg.m_MSNetworkCfg.ms_size;
    unsigned int dn_bw = this->stonne_cfg.m_SDMemoryCfg.n_read_ports;
    unsigned int rn_bw = this->stonne_cfg.m_SDMemoryCfg.n_write_ports;
    const char* output_directory=std::getenv("OUTPUT_DIR");
    std::string output_directory_str="";
    if(output_directory!=NULL) {
        std::string env_output_dir(output_directory);
        output_directory_str+=env_output_dir+"/";
    }

    out.open(output_directory_str+"output_stats_layer_"+this->dnn_layer->get_name()+"_architecture_MSes_"+std::to_string(num_ms)+"_dnbw_"+std::to_string(dn_bw)+"_"+"rn_bw_"+std::to_string(rn_bw)+"timestamp_"+std::to_string((int)time(NULL))+".txt"); //TODO Modify name somehow
    unsigned int indent=IND_SIZE;
    out << "{" << std::endl;

        //Printing input parameters
        this->stonne_cfg.printConfiguration(out, indent);
        out << "," << std::endl;

        //Printing layer configuration parameters
        this->dnn_layer->printConfiguration(out, indent);
        out << "," << std::endl;

        //Printing tile configuration parameters
        if (tile_loaded) {
            this->current_tile->printConfiguration(out, indent);
            out << "," << std::endl;
        }
        
        //Printing ASNetwork configuration parameters (i.e., ASwitches configuration for these VNs, flags, etc)
        this->asnet->printConfiguration(out, indent);
        out << "," << std::endl;
  
        this->msnet->printConfiguration(out, indent);
        out << "," << std::endl;

        
        //Printing global statistics
        this->printGlobalStats(out, indent);
        out << "," << std::endl;        

        //Printing all the components
        this->dsnet->printStats(out, indent);  //DSNetworkTop //DSNetworks //DSwitches
        out << "," << std::endl;
        this->msnet->printStats(out, indent);
        out << "," << std::endl;
        this->asnet->printStats(out, indent);
        out << "," << std::endl;
        this->mem->printStats(out, indent);
        out << "," << std::endl;
        this->collectionBus->printStats(out, indent);
        out << std::endl;
        
     
    
    out << "}" << std::endl;
    out.close();
}

void Stonne::printEnergy() {
    std::ofstream out;

    unsigned int num_ms = this->stonne_cfg.m_MSNetworkCfg.ms_size;
    unsigned int dn_bw = this->stonne_cfg.m_SDMemoryCfg.n_read_ports;
    unsigned int rn_bw = this->stonne_cfg.m_SDMemoryCfg.n_write_ports;

    const char* output_directory=std::getenv("OUTPUT_DIR");
    std::string output_directory_str="";
    if(output_directory!=NULL) {
        std::string env_output_dir(output_directory);
        output_directory_str+=env_output_dir+"/";
    }

    out.open(output_directory_str+"output_stats_layer_"+this->dnn_layer->get_name()+"_architecture_MSes_"+std::to_string(num_ms)+"_dnbw_"+std::to_string(dn_bw)+"_"+"rn_bw_"+std::to_string(rn_bw)+"timestamp_"+std::to_string((int)time(NULL))+".counters"); //TODO Modify name somehow
    unsigned int indent=0;
    out << "CYCLES=" <<  this->n_cycles << std::endl; //This is to calculate the static energy
    out << "[DSNetwork]" << std::endl;
    this->dsnet->printEnergy(out, indent);  //DSNetworkTop //DSNetworks //DSwitches
    out << "[MSNetwork]" << std::endl;
    this->msnet->printEnergy(out, indent);
    out << "[ReduceNetwork]" << std::endl;
    this->asnet->printEnergy(out, indent);
    out << "[GlobalBuffer]" << std::endl;
    this->mem->printEnergy(out, indent);
    out << "[CollectionBus]" << std::endl;
    this->collectionBus->printEnergy(out, indent);
    out << std::endl;

    out.close();

}

//Local function to the accelerator to print the globalStats
void Stonne::printGlobalStats(std::ofstream& out, unsigned int indent) {
    //unsigned int n_mswitches_used=this->current_tile->get_VN_Size()*this->current_tile->get_Num_VNs();
    //float percentage_mswitches_used = (float)n_mswitches_used / (float)this->stonne_cfg.m_MSNetworkCfg.ms_size;
    out << ind(indent) << "\"GlobalStats\" : {" << std::endl; //TODO put ID
    //out << ind(indent+IND_SIZE) << "\"N_mswitches_used\" : " << n_mswitches_used << "," << std::endl;
    //out << ind(indent+IND_SIZE) << "\"Percentage_mswitches_used\" : " << percentage_mswitches_used << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"N_cycles\" : " << this->n_cycles << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability

}

void Stonne::testMemory(unsigned int num_ms) {
   for(int i=0; i<20; i++) {
    this->mem->cycle();
    this->dsnet->cycle();
    this->msnet->cycle();
   }


    
}
void Stonne::testTile(unsigned int num_ms) {
    Tile* tile = new  Tile(3,1,1,2,1,1,1,1, false);
    //Tile* tile = new Tile(CONV, 2,2,1,2,1,2,2,1);
    //tile->generate_signals(num_ms);
    std::map<std::pair<int,int>, adderconfig_t> switches_configuration;// tile->get_switches_configuration();
    for(auto it=switches_configuration.begin(); it != switches_configuration.end(); ++it) {
        std::pair<int,int> current_node (it->first);
        adderconfig_t conf = it->second;
        std::cout << "Switch " << std::get<0>(current_node) << ":" << std::get<1>(current_node) << " --> " << get_string_adder_configuration(it->second) <<  std::endl;
    }
}

void Stonne::testDSNetwork(unsigned int num_ms) {
    //BRoadcast test
     /*
    DataPackage* data_to_send = new DataPackage(32, 1, IACTIVATION, 0, BROADCAST);
    std::vector<DataPackage*> vector_to_send;
    vector_to_send.push_back(data_to_send);
    this->inputConnection->send(vector_to_send);
    */

    //Unicast test
    /* 
    DataPackage* data_to_send = new DataPackage(32, 500, IACTIVATION, 0, UNICAST, 6);
    std::vector<DataPackage*> vector_to_send;
    vector_to_send.push_back(data_to_send);
    this->inputConnection->send(vector_to_send);
    */

    //Multicast test 
    
    bool* dests = new bool[num_ms]; //16 MSs
    for(int i=0;i<num_ms; i++) {
        dests[i]=false;
    }
    
    //Enabling Destinations 
    for(int i=0; i<6; i++)
        dests[i]=true;

    DataPackage* data_to_send = new DataPackage(32, 1, IACTIVATION, 0, MULTICAST, dests, num_ms);
    std::vector<DataPackage*> vector_to_send;
    vector_to_send.push_back(data_to_send);
    //this->inputDSConnection->send(vector_to_send);
    
    //Configuring the adders
    //First test
    std::map<std::pair<int,int>, adderconfig_t> switches_configuration; //Adders configuration
    std::map<std::pair<int,int>, fl_t> fwlinks_configuration;
    std::pair<int,int> switch0 (0,0);
    switches_configuration[switch0]=FW_2_2;

    std::pair<int,int> switch1(2,1);
    switches_configuration[switch1]=ADD_1_1_PLUS_FW_1_1;
    fwlinks_configuration[switch1]=SEND;

    std::pair<int,int> switch2(2,2);
    switches_configuration[switch2]=ADD_3_1;
    fwlinks_configuration[switch2]=RECEIVE;

//    asnet->addersConfiguration(switches_configuration);
 //   asnet->forwardingConfiguration(fwlinks_configuration);


 
    this->dsnet->cycle(); //TODO REVERSE THE ORDER!!!
    this->msnet->cycle();
    for(int i=0; i<7; i++) {
       this->lt->cycle();
       this->asnet->cycle(); // 2 to 1
    }
    
    delete[] dests;

}

