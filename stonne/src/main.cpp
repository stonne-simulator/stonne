#include <iostream>
#include "STONNEModel.h"
#include "types.h"
#include <chrono>
#include <assert.h>
#include "testbench.h"
#include <string>
#include <math.h>
#include <utility.h>

using namespace std;

void configConvParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &R, unsigned int &S, unsigned int &C, unsigned int &K, unsigned int &G, unsigned int &N, unsigned int &X, unsigned int &Y, unsigned int &strides,
                      unsigned int &T_R, unsigned int &T_S, unsigned int &T_C, unsigned int &T_K, unsigned int &T_G, unsigned int &T_N, unsigned int &T_X_, unsigned int &T_Y_, mRNA::OptGoal &mRNA_goal);

void configSparseGEMMParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &M, unsigned int &N, unsigned int &K, unsigned int &MK_sparsity, unsigned int &KN_sparsity, Dataflow &dataflow, unsigned int &optimize);

void configDenseGEMMParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &M, unsigned int &K, unsigned int &N, unsigned int &T_M, unsigned int &T_K, unsigned int &T_N, mRNA::OptGoal &mRNA_goal);

void configSparseDenseParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &M, unsigned int &N, unsigned int &K, unsigned int &MK_sparsity, unsigned int &T_N, unsigned int &T_K);

bool runConvCommand(int argc, char *argv[]);
bool runSparseGEMMCommand(int argc, char *argv[]);
bool runDenseGEMMCommand(int argc, char *argv[]);
bool runSparseDenseCommand(int argc, char *argv[]);
bool runHelpCommand();
//float* generateMatrixDense(unsigned int rows, unsigned int cols, unsigned int sparsity);

//void generateSparseDense(unsigned int rows, unsigned int cols, unsigned int sparsity);

//unsigned int* generateBitMapFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type);

float* generateMatrixSparseFromDense(float* denseMatrix, unsigned int* bitmap, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type);

//void generateSparseDense(unsigned int rows, unsigned int cols, unsigned int sparsity);
int* generateMinorIDFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, int &nnz, GENERATION_TYPE gen_type);
int* generateMajorPointerFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type);

void printDenseMatrix(float* matrix, unsigned int rows, unsigned int cols);
void printBitMap(unsigned int* bitmap, unsigned int rows, unsigned int cols);
void printSparseMatrix(float* sparseMatrix, unsigned int* bitmap, unsigned int rows, unsigned int cols);

int main(int argc, char *argv[]) {
    if(argc > 1) { //IF there is at least one parameter, -h is checked
        string arg = argv[1];
        if(arg=="-h") {
            runHelpCommand();
        }
      
        else if(arg=="-CONV") {
            runConvCommand(argc, argv);
        }

	else if(arg=="-SparseGEMM") {
            runSparseGEMMCommand(argc, argv);
	}

	else if((arg=="-DenseGEMM") || (arg=="-FC")) {
            runDenseGEMMCommand(argc, argv);
	}


	else {
	    std::cout << "How to use STONNE User Interface: ./" << argv[0] << " -h" << std::endl;
	}
    }

    else {
        std::cout << "How to use STONNE User Interface: ./" << argv[0] << " -h" << std::endl;
    }
}

bool runConvCommand(int argc, char *argv[]) {
    float EPSILON=0.05;
    unsigned int MAX_RANDOM=10; //Variable used to generate the random values
    /** Generating the inputs and outputs **/

    //Layer parameters (See MAERI paper to find out the taxonomy meaning)
    std::string layer_name="TestLayer";
    unsigned int R=1;                                  // R
    unsigned int S=3;                                  // S
    unsigned int C=1;                                  // C
    unsigned int K=1;                                  // K
    unsigned int G=1;                                  // G
    unsigned int N=1;                                  // N
    unsigned int X=1;                                  // X //TODO CHECK X=1 and Y=1
    unsigned int Y=3;                                  // Y
    unsigned int strides=1;                            // Strides
 
    //Tile parameters (See MAERI paper to find out the taxonomy meaning)
    unsigned int T_R=1;                                // T_R
    unsigned int T_S=3;                                // T_S
    unsigned int T_C=1;                                // T_C
    unsigned int T_K=1;                                // T_K
    unsigned int T_G=1;                                // T_G
    unsigned int T_N=1;                                // T_N
    unsigned int T_X_=1;                               // T_X
    unsigned int T_Y_=1;                               // T_Y

    //mRNA parameters
    mRNA::OptGoal mRNA_goal = mRNA::none;

    Config stonne_cfg; //Hardware parameters
//    stonne_cfg.m_MSNetworkCfg.ms_size=128;
    configConvParameters(argc, argv, stonne_cfg, layer_name, R, S, C, K, G, N, X, Y, strides, T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_, mRNA_goal); //Modify stonne_cfg and the variables according to user arguments

    //Calculating output parameters
    unsigned int X_= (X - R + strides) / strides;      // X_
    unsigned int Y_=(Y - S + strides) / strides;       // Y_
    std::cout << "Output Size: (X'=" << X_ << ", Y'=" << Y_ << std::endl; 


    //Creating arrays to store the ifmap ofmap and weights
    unsigned int ifmap_size=N*X*Y*C;
    unsigned int filter_size=R*S*(C/G)*K;
    unsigned int ofmap_size=N*X_*Y_*K;
    float* ifmap = new float[ifmap_size];
    float* filter = new float[filter_size];
    float* ofmap = new float[ofmap_size];
    float* ofmap_cpu = new float[ofmap_size]; //Used to store the CPU computed values to compare with the simulator version

    //Filling the arrays with random values
    for(int i=0; i<ifmap_size; i++) {
        ifmap[i]=rand()%MAX_RANDOM;
    }

    for(int i=0;i<filter_size; i++) {
        filter[i]=rand()%MAX_RANDOM;
    }

    //computing CPU version
    sequential_layer(R, S, C, K, G, N, X, Y, strides, ifmap, filter, ofmap_cpu); 

    /** END of generating the inputs and outputs **/
    //
    //
    //
    /** Configuring and running the accelerator  **/
    
    //Computing the CNN Layer with the simulator
    Stonne* stonne_instance = new Stonne(stonne_cfg); //Creating instance of the simulator
    stonne_instance->loadDNNLayer(CONV, layer_name, R, S, C, K, G, N, X, Y, strides, ifmap, filter, ofmap, CNN_DATAFLOW); //Loading the layer
    //Loads or generates a tile configuration depending on whether an mRNA goal has been specified
    if (mRNA_goal == mRNA::none)
        stonne_instance->loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_);
    else
        stonne_instance->generateConvTile(mRNA_goal);
    stonne_instance->run(); //Running the simulator 

    /** END of configuring and running the accelerator  **/
    //
    //
    //
    /** CHECKING the results to make sure that the output is correct  **/

    //Comparing the results
    for(int i=0;i<ofmap_size; i++) {
        float difference=fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference > EPSILON) {
            std::cout << "ERROR position " << i <<  ": Value ofmap simulator: " << ofmap[i] << ". Value ofmap CPU: " << ofmap_cpu[i] << std::endl;
            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
            delete[] ifmap;
            delete[] filter;
            delete[] ofmap;
            delete[] ofmap_cpu;
            delete stonne_instance;
            assert(false); //Always false
            
        }
    }


    //If the code does not stop then the TEST is correct
    std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;

    delete[] ifmap;
    delete[] filter;
    delete[] ofmap;
    delete[] ofmap_cpu;
    delete stonne_instance; 
    return true;
}

bool runDenseGEMMCommand(int argc, char *argv[]) {
    float EPSILON=0.05;
    unsigned int MAX_RANDOM=10; //Variable used to generate the random values
    /** Generating the inputs and outputs **/

    //Layer parameters (See MAERI paper to find out the taxonomy meaning)
    std::string layer_name="TestLayer";
    unsigned int M=1;                                  // M
    unsigned int K=3;                                  // K
    unsigned int N=1;                                  // N
 
    //Tile parameters 
    unsigned int T_M=1;                                // T_M
    unsigned int T_K=1;                                // T_K
    unsigned int T_N=1;                                // T_N

    //mRNA parameters
    mRNA::OptGoal mRNA_goal = mRNA::none;

    Config stonne_cfg; //Hardware parameters
//    stonne_cfg.m_MSNetworkCfg.ms_size=128;
    configDenseGEMMParameters(argc, argv, stonne_cfg, layer_name, M, K, N, T_M, T_K, T_N, mRNA_goal); //Modify stonne_cfg and the variables according to user arguments



    //Creating arrays to store the matrices
    unsigned int MK_size=M*K;
    unsigned int KN_size=N*K;
    unsigned int output_size=M*N;
    float* MK_matrix = new float[MK_size];
    float* KN_matrix = new float[KN_size];
    float* output = new float[output_size];
    float* output_cpu = new float[output_size]; //Used to store the CPU computed values to compare with the simulator version

    //Filling the arrays with random values
    for(int i=0; i<MK_size; i++) {
        MK_matrix[i]=rand()%MAX_RANDOM;
    }

    for(int i=0;i<KN_size; i++) {
        KN_matrix[i]=rand()%MAX_RANDOM;
    }


    //computing CPU version based on a Conv parameters mapping. Note a CONV layer might be seen as a GEMM if the mapping is correct.
    //sequential_layer(1, K, 1, M, 1, N, 1, K, 1, KN_matrix, MK_matrix, output_cpu); //Supposes that MK=inputs (M=batch size) and KN=filters (N=number of filters)
    //sequential_layer(1, K, 1, N, 1, 1, M, K, 1, MK_matrix, KN_matrix, output_cpu); //Supposes that MK=inputs (M=batch size) and KN=filters (N=number of filters)
    std::cout << "Printing MK matrix: " << std::endl;
    for(int i=0; i<M; i++) {
        for(int j=0; j<K; j++) {
            std::cout << MK_matrix[i*K+j] << " ";
        }
	std::cout << std::endl;
    
    }

    std::cout << "Printing KN matrix: " << std::endl;
    for(int i=0; i<N; i++) {
        for(int j=0; j<K; j++) {
            std::cout << KN_matrix[i*K+j] << " ";
        }
        std::cout << std::endl;

    }


    sequential_layer(1, K, 1, N, 1, M, 1, K, 1, MK_matrix, KN_matrix, output_cpu); //Supposes that MK=inputs (M=batch size) and KN=filters (N=number of filters)


    /** END of generating the inputs and outputs **/
    //
    //
    //
    /** Configuring and running the accelerator  **/

    //Computing the CNN Layer with the simulator
    Stonne* stonne_instance = new Stonne(stonne_cfg); //Creating instance of the simulator
    stonne_instance->loadDenseGEMM(layer_name, N, K, M, MK_matrix, KN_matrix, output, CNN_DATAFLOW); //Loading the layer
    //Loads or generates a tile configuration depending on whether an mRNA goal has been specified
    if (mRNA_goal == mRNA::none)
        stonne_instance->loadGEMMTile(T_N, T_K, T_M);
    else
        stonne_instance->generateFCTile(mRNA_goal);
    stonne_instance->run(); //Running the simulator 

    /** END of configuring and running the accelerator  **/
    //
    //
    //
    /** CHECKING the results to make sure that the output is correct  **/

    //Comparing the results
    for(int i=0;i<output_size; i++) {
        float difference=fabs(output[i]-output_cpu[i]);
        if(difference > EPSILON) {
            std::cout << "ERROR position " << i <<  ": Value ofmap simulator: " << output[i] << ". Value ofmap CPU: " << output_cpu[i] << std::endl;
            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
            delete[] MK_matrix;
            delete[] KN_matrix;
            delete[] output;
            delete[] output_cpu;
            delete stonne_instance;
            assert(false); //Always false
            
        }
    }


    //If the code does not stop then the TEST is correct
    std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;

    delete[] MK_matrix;
    delete[] KN_matrix;
    delete[] output;
    delete[] output_cpu;
    delete stonne_instance; 
    return true;
}

bool runSparseGEMMCommand(int argc, char *argv[]) {
    //Layer parameters (See SIGMA paper to find out the taxonomy meaning)
    float EPSILON=0.05;
    std::string layer_name="GEMMTestLayer";
    unsigned int M=4;                                  // M
    unsigned int N=4;                                  // N
    unsigned int K=8;                                  // K
    unsigned int optimize = 0; //False

    unsigned int MK_sparsity=20;
    unsigned int KN_sparsity=20;
    Dataflow dataflow = MK_STA_KN_STR;

    Config stonne_cfg;
    stonne_cfg.m_SDMemoryCfg.mem_controller_type=SIGMA_SPARSE_GEMM;

    configSparseGEMMParameters(argc, argv, stonne_cfg, layer_name, M, N, K, MK_sparsity, KN_sparsity, dataflow, optimize);

    //Creating MK matrix
    float* MK_dense_matrix_no_organized = generateMatrixDense(M, K, MK_sparsity);
    float* MK_dense_matrix = new float[M*K];
    
    //KN matrix
    float* KN_dense_matrix_no_organized = generateMatrixDense(K, N, KN_sparsity);
    float* KN_dense_matrix = new float[K*N];

    for(int i=0; i<M*K; i++) {
        MK_dense_matrix[i]=MK_dense_matrix_no_organized[i];
    }

    for(int i=0; i<K*N; i++) {
        KN_dense_matrix[i]=KN_dense_matrix_no_organized[i];

    }


    //See if it is necessary to reorganize
    unsigned int* order_table;
    if(optimize & (dataflow == MK_STA_KN_STR)) {


        order_table = calculateOrdering (MK_dense_matrix_no_organized, M,  K, GEN_BY_ROWS, stonne_cfg.m_MSNetworkCfg.ms_size);
        organizeMatrix (MK_dense_matrix, M, K, order_table, GEN_BY_ROWS);

    }

    else if(optimize & (dataflow==MK_STR_KN_STA)) {
        order_table = calculateOrdering (KN_dense_matrix_no_organized, K,  N, GEN_BY_COLS, stonne_cfg.m_MSNetworkCfg.ms_size);
        organizeMatrix (KN_dense_matrix, K, N, order_table, GEN_BY_COLS);


    }

    //Creating outputs
    float* cpu_output = new float[M*N];
    float* acc_output = new float[M*N];
    unsigned int* acc_bitmap = new unsigned int[M*N]; //Currently is not generated by the accelerator


    //Generating bitmaps
    unsigned int* MK_bitmap = generateBitMapFromDense(MK_dense_matrix, M, K, GEN_BY_ROWS);
    unsigned int* KN_bitmap = generateBitMapFromDense(KN_dense_matrix, K, N, GEN_BY_COLS);

    //Generating sparse matrix
    float* MK_sparse_matrix = generateMatrixSparseFromDense(MK_dense_matrix, MK_bitmap, M, K, GEN_BY_ROWS);
    float* KN_sparse_matrix = generateMatrixSparseFromDense(KN_dense_matrix, KN_bitmap, K, N, GEN_BY_COLS);

    //Running STONNE
    Stonne* stonne_instance = new Stonne(stonne_cfg); //Creating instance of the simulator
    stonne_instance->loadGEMM(layer_name, N, K, M, MK_sparse_matrix, KN_sparse_matrix, MK_bitmap, KN_bitmap, acc_output, acc_bitmap, dataflow ); //Loading GEMM
    stonne_instance->run(); //Running the simulator
    if(optimize && (dataflow==MK_STA_KN_STR)) {
        organizeMatrixBack (acc_output, M,  N, order_table, GEN_BY_ROWS);
    }

    else if(optimize && (dataflow==MK_STR_KN_STA)) {
        organizeMatrixBack(acc_output, M, N, order_table, GEN_BY_COLS);
    }

    /** CHECKING the results to make sure that the output is correct  **/
    std::cout << "Running CPU version to compare results" << std::endl;
    //Generating cpu output
    cpu_gemm(MK_dense_matrix_no_organized, KN_dense_matrix_no_organized, cpu_output, M, N, K);
/*
    std::cout << "Output matrix generated by CPU: " << std::endl;
    printDenseMatrix(cpu_output, M, N);
    std::cout << "Output matrix generated by STONNE: " << std::endl;
    printDenseMatrix(acc_output, M, N);
*/

    //Comparing the results
    for(int i=0;i<M; i++) {
        for(int j=0; j<N; j++) {
            float difference=fabs(cpu_output[i*N+j]-acc_output[i*N+j]);
            if(difference > EPSILON) {
                std::cout << "ERROR position (" << i << "," << j <<  "): Value out simulator: " << acc_output[i*N+j] << ". Value out CPU: " << cpu_output[i*N+j] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
		delete[] MK_dense_matrix;
                delete[] KN_dense_matrix;
		delete[] MK_dense_matrix_no_organized;
		delete[] KN_dense_matrix_no_organized;
                delete[] MK_bitmap;
                delete[] KN_bitmap;
                delete[] MK_sparse_matrix;
                delete[] KN_sparse_matrix;
                delete[] cpu_output;
                delete[] acc_output;
                delete[] acc_bitmap;
		if(optimize)
		    delete[] order_table;
                delete stonne_instance;

                assert(false); //Always false

            }

        }
    }


    //If the code does not stop then the TEST is correct
    std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;


    delete[] MK_dense_matrix_no_organized;
    delete[] KN_dense_matrix_no_organized;
    delete[] MK_dense_matrix;
    delete[] KN_dense_matrix;
    delete[] MK_bitmap;
    delete[] KN_bitmap;
    delete[] MK_sparse_matrix;
    delete[] KN_sparse_matrix;
    delete[] cpu_output;
    delete[] acc_output;
    delete[] acc_bitmap;
    if(optimize)
        delete[] order_table;
    
    delete stonne_instance;
    return true;    



}

bool runHelpCommand() {
    std::cout << "Welcome to the STONNE User Interface Version 1.0: " << std::endl;
    std::cout << "***********************************************************************************************************" << std::endl;
    std::cout << "***********************************************************************************************************" << std::endl << std::endl;
    std::cout << "The STONNE user interface allows to execute the STONNE simulator with any parameter. Currently, STONNE runs" << std::endl;
    std::cout << "4 types of operations: Convolution Layers, FC Layers, Dense GEMMs and Sparse GEMMs. Note that almost any" << std::endl;
    std::cout << "kernel can be, in the end, mapped using these operations." << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: ./stonne [-h | -CONV | -FC | -SparseGEMM | -DenseGEMM] [Hardware Parameters] [Dimension and tile Parameters]"  << std::endl;
    std::cout  << std::endl;
    std::cout << "[Hardware parameters]" << std::endl;
    std::cout << "-num_ms=[x]: Number of multiplier switches (must be power of 2)" << std::endl;
    std::cout << "-dn_bw=[x]: Number of read ports in the SDMemory (must be power of 2)" << std::endl;
    std::cout << "-rn_bw=[x]: Number of write ports in the SDMemory (must be power of 2)" << std::endl;
    std::cout << "-rn_type=[0=ASNETWORK, 1=FENETWORK]: type of the ReduceNetwork to be used (Not supported for SparseGEMM)" << std::endl;
    std::cout << "-accumulation_buffer=[0,1]: enables the accumulation buffer" << std::endl;
    std::cout << "-print_stats=[0,1]: Flag that enables the printing of the statistics" << std::endl;
    std::cout << std::endl;

    std::cout << "[Dimension and Tile parameters]" << std::endl;
    std::cout << "The dimensions of the kernel depends on the type of the operation that is going to be run. Next, we describe" << std::endl;
    std::cout << "described the dimensions according to each supported operation." << std::endl;
    std::cout << std::endl;
    std::cout << "[CONV]" << std::endl;
    std::cout << "-layer_name: Name of the layer to run. The output statistic file will be named accordingly" << std::endl; 
    std::cout << "-R: Number of flter rows" << std::endl;
    std::cout << "-S: Number of filter columns" << std::endl;
    std::cout << "-C: Number of filter and input channels" << std::endl;
    std::cout << "-K: Number of filters and output channels" << std::endl;
    std::cout << "-G: Number of groups" << std::endl;
    std::cout << "-N: Number of inputs (Only 1 is supported so far)" << std::endl;
    std::cout << "-X: Number of input rows" << std::endl;
    std::cout << "-Y: Number of input columns" << std::endl;
    std::cout << "-strides: Stride value used in the layer" << std::endl;
    std::cout << "-T_R: Number of flter rows mapped at a time" << std::endl;
    std::cout << "-T_S: Number of filter columns mapped at a time" << std::endl;
    std::cout << "-T_C: Number of filter and input channels per group mapped at a time" << std::endl;
    std::cout << "-T_K: Number of filters and output channels per group mapped at a time" << std::endl;
    std::cout << "-T_G: Number of groups mappd at a time" << std::endl;
    std::cout << "-T_N: Number of inputs mapped at a time (Only 1 is supported so far)" << std::endl;
    std::cout << "-T_X_: Number of input rows mapped at a time" << std::endl;
    std::cout << "-T_Y_: Number of input columns mapped a time" << std::endl;
    std::cout << "** Please take into consideration that: **" << std::endl;
    std::cout << "1. Number of Virtual Neurons mapped (Num_VNs) will be T_K*T_G*T_N*T_X_*T_T_" << std::endl;
    std::cout << "2. The minimum number of MSwitches needed will be at least VN_Size*Num_VNs" << std::endl;
    std::cout << "3. Note in case of folding (iteration over the same VN) is enabled, and the accumulation buffer disabled" << std::endl;
    std::cout << "1 extra MSwitch per VN will be needed to manage the psum." << std::endl;
    std::cout << "In this case, the minimum number of MSwitches needed will be at least (VN_Size+1)*Num_VNs" << std::endl;
    std::cout << "Folding (iteration over the same virtual neuron) will be enabled if (R/T_S)*(S/T_S)*(C/T_C) > 1" << std::endl;           std::cout << std::endl;
    std::cout << "[FC]" << std::endl;
    std::cout << "-layer_name: Name of the layer to run. The output statistic file will be called by this name" << std::endl;
    std::cout << "-M=Number of output neurons" << std::endl;
    std::cout << "-N=Batch size" << std::endl;
    std::cout << "-K=Number of input neurons" << std::endl;
    std::cout << "-T_M=Number of output neurons mapped at a time" << std::endl;
    std::cout << "-T_N=Number of batches mapped at a time" << std::endl;
    std::cout << "-T_K=Number of input neurons mapped at a time" << std::endl;
    std::cout << "** Please take into consideration that: **" << std::endl;
    std::cout << "1. Number of Virtual Neurons mapped (Num_VNs) will be T_N*T_M" << std::endl;
    std::cout << "2. The minimum number of MSwitches needed will be at least T_K*Num_VNs." << std::endl;
    std::cout << std::endl;
    std::cout << "[DenseGEMM]" << std::endl;
    std::cout << "-layer_name: Name of the layer to run. The output statistic file will be called by this name" << std::endl;
    std::cout << "-M=Number of rows MK matrix" << std::endl;
    std::cout << "-N=Number of columns KN matrix" << std::endl;
    std::cout << "-K=Number of columns MK and rows KN matrix (cluster size)" << std::endl;
    std::cout << "-T_M=Number of M rows mapped at a time" << std::endl;
    std::cout << "-T_N=Number of N columns at a time" << std::endl;
    std::cout << "-T_K=Number of K elements mapped at a time" << std::endl;
    std::cout << "** Please take into consideration that: **" << std::endl;
    std::cout << "1. Number of Virtual Neurons mapped (Num_VNs) will be T_N*T_M" << std::endl;
    std::cout << "2. The minimum number of MSwitches needed will be at least T_K*Num_VNs." << std::endl;
    std::cout << std::endl;
    std::cout << "[SparseGEMM]" << std::endl;
    std::cout << "-layer_name: Name of the layer to run. The output statistic file will be called by this name" << std::endl;
    std::cout << "-M=Number of rows MK matrix" << std::endl;
    std::cout << "-N=Number of columns KN matrix" << std::endl;
    std::cout << "-K=Number of columns MK and rows KN matrix (cluster size)" << std::endl;
    std::cout << "-MK_sparsity=Percentage of sparsity MK matrix (0-100)" << std::endl;
    std::cout << "-KN_sparsity=Percentahe of sparsity KN matrix (0-100)" << std::endl;
    std::cout << "-dataflow=MK_STA_KN_STR or MK_STR_KN_STA" << std::endl;
    std::cout << "-optimize=[0,1]: apply compiler-based optimizations" << std::endl;

    exit(0);
    return true; //Never executed
}

//This function modifies the default values of the parameters according to user arguments.
void configConvParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &R, unsigned int &S, unsigned int &C, unsigned int &K, unsigned int &G, unsigned int &N, unsigned int &X, unsigned int &Y, unsigned int &strides,
                      unsigned int &T_R, unsigned int &T_S, unsigned int &T_C, unsigned int &T_K, unsigned int &T_G, unsigned int &T_N, unsigned int &T_X_, unsigned int &T_Y_, mRNA::OptGoal &mRNA_goal) {

    //Parsing
    for(int i=2; i<argc; i++) { //0 is the name of the program and 1 is the execution command type
        string arg = argv[i];
        //Spliting using = character
        string::size_type pos = arg.find('=');
        if(arg.npos != pos) {
            string value_str=arg.substr(pos+1);
            string name=arg.substr(0, pos);
            unsigned int value;
            if((name != "-layer_name") && (name != "-rn_type")) { //string parameters
                value=stoi(value_str);
            }
            //Checking parameter name
            if(name=="-num_ms") {
                if(!ispowerof2(value)) {   //Checking that the num_ms is power of 2
                    std::cout << "Error: -num_ms must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing num_ms to " << value << std::endl; //To debug
                stonne_cfg.m_MSNetworkCfg.ms_size=value;
            }

            else if(name=="-dn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -dn_bw must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing dn_bw to " << value << std::endl; //To debug
                stonne_cfg.m_SDMemoryCfg.n_read_ports=value;
            }

            else if(name=="-rn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -rn_bw must be power of 2" << std::endl; 
                    exit(1);
                }
                std::cout << "Changing rn_bw to " << value << std::endl;
                stonne_cfg.m_SDMemoryCfg.n_write_ports=value;
            }

	    else if(name=="-accumulation_buffer") {
                std::cout << "Changing accumulation_buffer to " << value << std::endl;
                stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled=value;
            }




            else if(name=="-print_stats") {
                if((value != 0) && (value != 1)) {
                    std::cout << "Error: -print_stats only supports 0 or 1" << std::endl;
                    exit(1);
                }
                std::cout << "Changing print_stats to " << value << std::endl;
                stonne_cfg.print_stats_enabled=value; 
            }

            else if(name=="-rn_type") {
                std::cout << "Changing rn_type to " << value_str << std::endl;
                stonne_cfg.m_ASNetworkCfg.reduce_network_type=get_type_reduce_network_type(value_str);
            }

            //Running configuration parameters (layer and tile)
   
           //Layer parameters
           else if(name=="-layer_name") {
               std::cout << "Changing layer_name to " << value_str << std::endl;
               layer_name=value_str; 
           }
        
           else if(name=="-R") {
                std::cout << "Changing R to " << value << std::endl;
                R=value;
           }

           else if(name=="-S") {
                std::cout << "Changing S to " << value << std::endl;
                S=value;
           }

           else if(name=="-C") {
                std::cout << "Changing C to " << value << std::endl;
                C=value;
           } 
        
           else if(name=="-K") {
                std::cout << "Changing K to " << value << std::endl;
                K=value;
           }
  
           else if(name=="-G") {
               std::cout << "Changing G to " << value << std::endl;
               G=value;
           }
  
           else if(name=="-N") {
                std::cout << "Changing N to " << value << std::endl;
                N=value;
           }

           else if(name=="-X") {
                std::cout << "Changing X to " << value << std::endl;
                X=value;
           }

           else if(name=="-Y") {
                std::cout << "Changing Y to " << value << std::endl;
                Y=value;
           }

           else if(name=="-strides") {
               std::cout << "Changing strides to " << value << std::endl;
               strides=value;
           }

           //Tile parameters
           else if(name=="-T_R") {
                std::cout << "Changing T_R to " << value << std::endl;
                T_R=value;
           } 

           else if(name=="-T_S") {
                std::cout << "Changing T_S to " << value << std::endl;
                T_S=value;
           }

           else if(name=="-T_C") {
                std::cout << "Changing T_C to " << value << std::endl;
                T_C=value;
           }

           else if(name=="-T_K") {
                std::cout << "Changing T_K to " << value << std::endl;
                T_K=value;
           }

           else if(name=="-T_G") {
               std::cout << "Changing T_G to " << value << std::endl;
               T_G=value;
           }

           else if(name=="-T_N") {
                std::cout << "Changing T_N to " << value << std::endl;
                T_N=value;
           }

           else if(name=="-T_X_") {
                std::cout << "Changing T_X_ to " << value << std::endl;
                T_X_=value;
           }

           else if(name=="-T_Y_") {
                std::cout << "Changing T_Y_ to " << value << std::endl;
                T_Y_=value;
           }

            else if(name=="-mRNA") {
                if((value < 0) || (value > 3)) {
                    std::cout << "Error: -mRNA only supports 0 (none), 1 (performance), 2 (energy) or 3 (energy_efficiency)" << std::endl;
                    exit(1);
                }
                std::cout << "Changing mRNA to " << value << std::endl;
                mRNA_goal = parsemRNAGoal(to_string(value));
            }

           //Parameter is not recognized
           else {
                std::cout << "Error: parameter " << name << " does not exist" << std::endl;
                exit(1);
            }

 
    
           

        }
        else {

            std::cout << "Error: parameter " << arg << " does not exist" << std::endl;
            exit(1);

        }
    }
}

void configDenseGEMMParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &M, unsigned int &K, unsigned int &N, unsigned int &T_M, unsigned int &T_K, unsigned int &T_N, mRNA::OptGoal &mRNA_goal) {
  //Parsing
    for(int i=2; i<argc; i++) { //0 is the name of the program and 1 is the execution command type
        string arg = argv[i];
        //Spliting using = character
        string::size_type pos = arg.find('=');
        if(arg.npos != pos) {
            string value_str=arg.substr(pos+1);
            string name=arg.substr(0, pos);
            unsigned int value;
            if((name != "-layer_name") && (name != "-rn_type") && (name != "-mn_type") && (name != "-mem_ctrl")) { //string parameters
                value=stoi(value_str);
            }
            //Checking parameter name
            if(name=="-num_ms") {
                if(!ispowerof2(value)) {   //Checking that the num_ms is power of 2
                    std::cout << "Error: -num_ms must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing num_ms to " << value << std::endl; //To debug
                stonne_cfg.m_MSNetworkCfg.ms_size=value;
            }

	    else if(name=="-ms_rows") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -ms_rows must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing ms_rows to " << value << std::endl; //To debug
                stonne_cfg.m_MSNetworkCfg.ms_rows=value;

            }


	    else if(name=="-ms_cols") {
                if(!ispowerof2(value)) {   
                    std::cout << "Error: -ms_cols must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing ms_cols to " << value << std::endl; //To debug
                stonne_cfg.m_MSNetworkCfg.ms_cols=value;

	    }


            else if(name=="-dn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -dn_bw must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing dn_bw to " << value << std::endl; //To debug
                stonne_cfg.m_SDMemoryCfg.n_read_ports=value;
            }

            else if(name=="-rn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -rn_bw must be power of 2" << std::endl; 
                    exit(1);
                }
                std::cout << "Changing rn_bw to " << value << std::endl;
                stonne_cfg.m_SDMemoryCfg.n_write_ports=value;
            }

	     else if(name=="-accumulation_buffer") {
                std::cout << "Changing accumulation_buffer to " << value << std::endl;
                stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled=value;
            }

            else if(name=="-print_stats") {
                if((value != 0) && (value != 1)) {
                    std::cout << "Error: -print_stats only supports 0 or 1" << std::endl;
                    exit(1);
                }
                std::cout << "Changing print_stats to " << value << std::endl;
                stonne_cfg.print_stats_enabled=value; 
            }

	     else if(name=="-rn_type") {
                std::cout << "Changing rn_type to " << value_str << std::endl;
                stonne_cfg.m_ASNetworkCfg.reduce_network_type=get_type_reduce_network_type(value_str);
            }

	    else if(name=="-mn_type") {
                std::cout << "Changing mn_type to " << value_str << std::endl;
                stonne_cfg.m_MSNetworkCfg.multiplier_network_type=get_type_multiplier_network_type(value_str);
            }

	    else if(name=="-mem_ctrl") {
                std::cout << "Changing mem_ctrl to " << value_str << std::endl;
                stonne_cfg.m_SDMemoryCfg.mem_controller_type=get_type_memory_controller_type(value_str);
            }

            //Running configuration parameters (layer)
   
           //Layer parameters
           else if(name=="-layer_name") {
               std::cout << "Changing layer_name to " << value_str << std::endl;
               layer_name=value_str; 
           }
        
           else if(name=="-M") {
                std::cout << "Changing M to " << value << std::endl;
                M=value;
           }

           else if(name=="-N") {
                std::cout << "Changing N to " << value << std::endl;
                N=value;
           }

           else if(name=="-K") {
                std::cout << "Changing K to " << value << std::endl;
                K=value;
           } 
        
           else if(name=="-T_M") {
                std::cout << "Changing T_M to " << value << std::endl;
                T_M=value;
           }
  
           else if(name=="-T_N") {
               std::cout << "Changing T_N to " << value << std::endl;
               T_N=value;
           }
  
	        else if(name=="-T_K") {
               std::cout << "Changing T_K to " << value << std::endl;
               T_K=value;
           }

            else if(name=="-mRNA") {
                if((value < 0) || (value > 3)) {
                    std::cout << "Error: -mRNA only supports 0 (none), 1 (performance), 2 (energy) or 3 (energy_efficiency)" << std::endl;
                    exit(1);
                }
                std::cout << "Changing mRNA to " << value << std::endl;
                mRNA_goal = parsemRNAGoal(to_string(value));
            }

           //Parameter is not recognized
           else {
                std::cout << "Error: parameter " << name << " does not exist" << std::endl;
                exit(1);
            }

 
    
           

        }
        else {

            std::cout << "Error: parameter " << arg << " does not exist" << std::endl;
            exit(1);

        }
    }  

}



void configSparseGEMMParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &M, unsigned int &N, unsigned int &K, unsigned int &MK_sparsity, unsigned int &KN_sparsity, Dataflow &dataflow, unsigned int &optimize) {
    //Parsing
    for(int i=2; i<argc; i++) { //0 is the name of the program and 1 is the execution command type
        string arg = argv[i];
        //Spliting using = character
        string::size_type pos = arg.find('=');
        if(arg.npos != pos) {
            string value_str=arg.substr(pos+1);
            string name=arg.substr(0, pos);
            unsigned int value;
            if((name != "-layer_name") && (name != "-dataflow")) { //string parameters
                value=stoi(value_str);
            }
            //Checking parameter name
            if(name=="-num_ms") {
                if(!ispowerof2(value)) {   //Checking that the num_ms is power of 2
                    std::cout << "Error: -num_ms must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing num_ms to " << value << std::endl; //To debug
                stonne_cfg.m_MSNetworkCfg.ms_size=value;
            }

            else if(name=="-dn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -dn_bw must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing dn_bw to " << value << std::endl; //To debug
                stonne_cfg.m_SDMemoryCfg.n_read_ports=value;
            }

            else if(name=="-rn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -rn_bw must be power of 2" << std::endl; 
                    exit(1);
                }
                std::cout << "Changing rn_bw to " << value << std::endl;
                stonne_cfg.m_SDMemoryCfg.n_write_ports=value;
            }



            else if(name=="-print_stats") {
                if((value != 0) && (value != 1)) {
                    std::cout << "Error: -print_stats only supports 0 or 1" << std::endl;
                    exit(1);
                }
                std::cout << "Changing print_stats to " << value << std::endl;
                stonne_cfg.print_stats_enabled=value; 
            }

            //Running configuration parameters (layer)
   
           //Layer parameters
           else if(name=="-layer_name") {
               std::cout << "Changing layer_name to " << value_str << std::endl;
               layer_name=value_str; 
           }
        
           else if(name=="-M") {
                std::cout << "Changing M to " << value << std::endl;
                M=value;
           }

           else if(name=="-N") {
                std::cout << "Changing N to " << value << std::endl;
                N=value;
           }

           else if(name=="-K") {
                std::cout << "Changing K to " << value << std::endl;
                K=value;
           } 
        
           else if(name=="-MK_sparsity") {
                std::cout << "Changing MK_sparsity to " << value << std::endl;
                MK_sparsity=value;
           }
  
           else if(name=="-KN_sparsity") {
               std::cout << "Changing KN_sparsity to " << value << std::endl;
               KN_sparsity=value;
           }
  
	   else if(name=="-dataflow") {
                std::cout << "Changing dataflow to " << value_str << std::endl;
                dataflow=get_type_dataflow_type(value_str);
            }

	    else if(name=="-optimize") {
               std::cout << "Changing optimize " << value << std::endl;
               optimize=value;
           }



           //Parameter is not recognized
           else {
                std::cout << "Error: parameter " << name << " does not exist" << std::endl;
                exit(1);
            }

 
    
           

        }
        else {

            std::cout << "Error: parameter " << arg << " does not exist" << std::endl;
            exit(1);

        }
    }
}

void configSparseDenseParameters(int argc, char *argv[], Config &stonne_cfg, std::string &layer_name, unsigned int &M, unsigned int &N, unsigned int &K, unsigned int &MK_sparsity, unsigned int &T_N, unsigned int &T_K) {
    //Parsing
    for(int i=2; i<argc; i++) { //0 is the name of the program and 1 is the execution command type
        string arg = argv[i];
        //Spliting using = character
        string::size_type pos = arg.find('=');
        if(arg.npos != pos) {
            string value_str=arg.substr(pos+1);
            string name=arg.substr(0, pos);
            unsigned int value;
            if((name != "-layer_name") && (name != "-rn_type")) { //string parameters
                value=stoi(value_str);
            }
            //Checking parameter name
            if(name=="-num_ms") {
                if(!ispowerof2(value)) {   //Checking that the num_ms is power of 2
                    std::cout << "Error: -num_ms must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing num_ms to " << value << std::endl; //To debug
                stonne_cfg.m_MSNetworkCfg.ms_size=value;
            }

            else if(name=="-dn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -dn_bw must be power of 2" << std::endl;
                    exit(1);
                }
                std::cout << "Changing dn_bw to " << value << std::endl; //To debug
                stonne_cfg.m_SDMemoryCfg.n_read_ports=value;
            }

            else if(name=="-rn_bw") {
                if(!ispowerof2(value)) {
                    std::cout << "Error: -rn_bw must be power of 2" << std::endl; 
                    exit(1);
                }
                std::cout << "Changing rn_bw to " << value << std::endl;
                stonne_cfg.m_SDMemoryCfg.n_write_ports=value;
            }



            else if(name=="-print_stats") {
                if((value != 0) && (value != 1)) {
                    std::cout << "Error: -print_stats only supports 0 or 1" << std::endl;
                    exit(1);
                }
                std::cout << "Changing print_stats to " << value << std::endl;
                stonne_cfg.print_stats_enabled=value; 
            }

            //Running configuration parameters (layer)
   
           //Layer parameters
           else if(name=="-layer_name") {
               std::cout << "Changing layer_name to " << value_str << std::endl;
               layer_name=value_str; 
           }
        
           else if(name=="-M") {
                std::cout << "Changing M to " << value << std::endl;
                M=value;
           }

           else if(name=="-N") {
                std::cout << "Changing N to " << value << std::endl;
                N=value;
           }

           else if(name=="-K") {
                std::cout << "Changing K to " << value << std::endl;
                K=value;
           } 
        
           else if(name=="-MK_sparsity") {
                std::cout << "Changing MK_sparsity to " << value << std::endl;
                MK_sparsity=value;
           }
  
           else if(name=="-T_N") {
               std::cout << "Changing T_N to " << value << std::endl;
               T_N=value;
           }

	    else if(name=="-T_K") {
               std::cout << "Changing T_K to " << value << std::endl;
               T_K=value;
           }
           
           else if(name=="-accumulation_buffer") {
                std::cout << "Changing accumulation_buffer to " << value << std::endl;
                stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled=value;
            }

	   else if(name=="-rn_type") {
                std::cout << "Changing rn_type to " << value_str << std::endl;
                stonne_cfg.m_ASNetworkCfg.reduce_network_type=get_type_reduce_network_type(value_str);
            }

           //Parameter is not recognized
           else {
                std::cout << "Error: parameter " << name << " does not exist" << std::endl;
                exit(1);
            }

    
           

        }
        else {

            std::cout << "Error: parameter " << arg << " does not exist" << std::endl;
            exit(1);

        }
    }
}


