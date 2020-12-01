#include <iostream>
#include "DNNLayer.h"
#include "STONNEModel.h"
#include <assert.h>
#include "types.h"


void sequential_layer(unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, 
float* input, float* filters, float * outputs) {

    unsigned int OX=(X - R + strides) / strides;
    unsigned int OY=(Y - S + strides) / strides;
    K=K/G;
    C=C/G;
    unsigned int output_size_n = G*K*OX*OY;
    unsigned int input_size_n = G*C*X*Y;
    unsigned int filter_size=R*S*C;
    unsigned int size_oy=OY*K*G;
    unsigned int size_y=Y*G*C;
    for(int n=0; n<N; n++) {
        for(int g=0; g<G; g++) {
            for(int k=0; k<K; k++) {
                for(int ox=0; ox<OX; ox++) {
                    for(int oy=0; oy<OY; oy++) {
                        outputs[n*output_size_n + ox*size_oy + oy*K*G + g*K + k]=0.0;
                        for(int c=0; c<C;c++) {
                            for(int r=0;r<R;r++) {
                                for(int s=0;s<S;s++) {
                                    outputs[n*output_size_n + ox*size_oy + oy*K*G + g*K + k] += input[n*input_size_n+ ox*strides*size_y + oy*strides*C*G + r*size_y + s*C*G + g*C + c]*filters[g*K*filter_size + k*filter_size + r*S*C + s*C + c];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

void cpu_gemm(float* MK_dense_matrix, float* KN_dense_matrix, float* output, unsigned int M, unsigned int N, unsigned int K) {
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
	    float suma=0;
            for(int k=0; k<K; k++) {
                suma+=MK_dense_matrix[i*K+k]*KN_dense_matrix[k*N+j];
	    }

	    output[i*N+j]=suma;
	}
    }
}


bool run_single_test_cnn(DNNLayer* dnn_layer, unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G, unsigned int T_N, 
unsigned int T_X_, unsigned int T_Y_, Config stonne_cfg, unsigned int n_test) {
    float EPSILON=0.001;
    unsigned int MAX_RANDOM=100; //Variable used to generate the random values
    //Extracting the parameters 
    unsigned int R=dnn_layer->get_R();                 // R
    unsigned int S=dnn_layer->get_S();                 // S
    unsigned int C=dnn_layer->get_C();                 // C
    unsigned int K=dnn_layer->get_K();                 // K
    unsigned int G=dnn_layer->get_G();
    unsigned int N=dnn_layer->get_N();                 // N
    unsigned int X=dnn_layer->get_X();                 // X
    unsigned int Y=dnn_layer->get_Y();                 // Y
    unsigned int strides=dnn_layer->get_strides();     // strides
    unsigned int X_=dnn_layer->get_X_();               // X_
    unsigned int Y_=dnn_layer->get_Y_();               // Y_
    //Creating arrays to store the ifmap ofmap and weights
    unsigned int ifmap_size=X*Y*C;
    unsigned int filter_size=R*S*(C/G)*K;
    unsigned int ofmap_size=X_*Y_*K;
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

    //Computing the CNN Layer with the simulator
    Stonne* stonne_instance = new Stonne(stonne_cfg); //Creating instance of the simulator
    stonne_instance->loadDNNLayer(CONV, "TestLayer", R, S, C, K, G, N, X, Y, strides, ifmap, filter, ofmap, CNN_DATAFLOW); //Loading the layer
    stonne_instance->loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_); //Loading the tile
    stonne_instance->run(); //Running the simulator 
    
    //Comparing the results
    for(int i=0;i<ofmap_size; i++) {
        float difference=abs(ofmap[i]-ofmap_cpu[i]);
        if(difference > EPSILON) {
    //        std::cout << "ERROR position " << i <<  ": Value ofmap simulator: " << ofmap[i] << ". Value ofmap CPU: " << ofmap_cpu[i] << std::endl;
            std::cout << "\033[1;31m[ERROR] \033[0m" << "Test " << n_test << " Failed: " << "HARDWARE_PARAMETERS=(num_ms=" << stonne_cfg.m_MSNetworkCfg.ms_size  << ") LAYER=(" << "R=" << R << ", S=" << S << ", C=" << C << ", K=" << K << ", N=" << N << ", X=" << X << ", Y=" << Y << ", strides=" << strides << ", X_=" << X_ << ", Y_=" << Y_ << ") TILE=(" << "T_R=" << T_R << ", T_S=" << T_S << ", T_C=" << T_C << ", T_K=" << T_K << ", T_N=" << T_N << ", T_X_=" << T_X_ << ", T_Y_=" << T_Y_ << ")" << std::endl; //by the moment we just print num_ms as hardware parameters
            std::cout << "\033[1;31mTests not passed\033[0m" << std::endl;
            delete[] ifmap;
            delete[] filter;
            delete[] ofmap;
            delete[] ofmap_cpu;
            delete stonne_instance;
            assert(false); //Always false
            
        }
    }


    //If the code does not stop then the TEST is correct
    std::cout << "\033[1;32m[OK] \033[0m" <<  "Test " << n_test << " passed: " << "HARDWARE_PARAMETERS=(num_ms=" << stonne_cfg.m_MSNetworkCfg.ms_size  << ") LAYER=(" << "R=" << R << ", S=" << S << ", C=" << C << ", K=" << K << ", N=" << N << ", X=" << X << ", Y=" << Y << ", strides=" << strides << ", X_=" << X_ << ", Y_=" << Y_ << ") TILE=(" << "T_R=" << T_R << ", T_S=" << T_S << ", T_C=" << T_C << ", T_K=" << T_K << ", T_N=" << T_N << ", T_X_=" << T_X_ << ", T_Y_=" << T_Y_ << ")" << std::endl; //by the moment we just print num_ms as hardware parameters

    delete[] ifmap;
    delete[] filter;
    delete[] ofmap;
    delete[] ofmap_cpu;
    delete stonne_instance; 
    return true;
    
        
}


void run_real_tests() {
     std::cout << "Executing real tests" << std::endl;
     Config stonne_cfg; //Creating stonne configuration object

    // Creating layer configuration
    unsigned int R = 3;
    unsigned int S = 3;
    unsigned int C = 256;
    unsigned int K = 384;
    unsigned int G = 1;
    unsigned int N = 1;
    unsigned int X = 6;
    unsigned int Y = 6;
    unsigned int strides = 1;

    DNNLayer* dnn_layer = new DNNLayer(CONV, "RealTestLayer",R,S,C,K,G,N,X,Y,strides);


    unsigned int T_R, T_S, T_C, T_K, T_G, T_N, T_X, T_Y;

   // Hardware configuration 1 (ms=32) 
   stonne_cfg.m_MSNetworkCfg.ms_size=32;  
   for(int i=1; i<32; i=i*2) {
       stonne_cfg.m_SDMemoryCfg.n_read_ports=i;
       stonne_cfg.m_SDMemoryCfg.n_write_ports=i;
       //VN Size = 32
       T_R=1; T_S=1; T_C=32; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=1
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1

       //VN Size = 18
       T_R=3; T_S=3; T_C=2; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=1
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1


 

       //VN Size = 16
       T_R=1; T_S=1; T_C=16; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=1
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=1; T_S=1; T_C=16; T_K=2; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=1; T_S=1; T_C=16; T_K=1; T_G=1; T_N=1; T_X=2; T_Y=1; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=1; T_S=1; T_C=16; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=2; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G, T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1

       //VN Size = 12
       T_R=3; T_S=1; T_C=4; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=1
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G, T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=3; T_S=1; T_C=4; T_K=2; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=3; T_S=1; T_C=4; T_K=1; T_G=1; T_N=1; T_X=2; T_Y=1; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=3; T_S=1; T_C=4; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=2; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1

       T_R=1; T_S=3; T_C=4; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=1
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=1; T_S=3; T_C=4; T_K=2; T_G=1; T_N=1; T_X=1; T_Y=1; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G, T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=1; T_S=3; T_C=4; T_K=1; T_G=1; T_N=1; T_X=2; T_Y=1; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1
       T_R=1; T_S=3; T_C=4; T_K=1; T_G=1; T_N=1; T_X=1; T_Y=2; //NumVNs=2
       run_single_test_cnn(dnn_layer, T_R,  T_S,  T_C,  T_K, T_G,  T_N,  T_X,  T_Y, stonne_cfg,1); // Executes test 1

       

  








   }
   

  // K TEST
   run_single_test_cnn(dnn_layer, R,  S,  1,  2, 1,  1,  1,  1, stonne_cfg,7);
   run_single_test_cnn(dnn_layer, R,  S,  1,  2, 1,  1,  1,  1, stonne_cfg,8);
   std::cout << "\033[1;32mAll tests have been passed successfully\033[0m\n" << std::endl;
   delete dnn_layer;




}

void run_simple_tests() {
   std::cout << "Executing simple tests" << std::endl;
   //Hardware parameters by default. TODO modify according the new parameters added 
   Config stonne_cfg;

  //Simple layer 
   unsigned int R = 3;
   unsigned int S = 3;
   unsigned int C = 6;
   unsigned int K = 6;
   unsigned int G = 1;
   unsigned int N = 1;
   unsigned int X = 5;
   unsigned int Y = 5;
   unsigned int strides = 1;
   DNNLayer* dnn_layer = new DNNLayer(CONV, "simpleTestName", R,S,C,K,G, N,X,Y,strides);
   
   //Tests varying the tile      T_R T_S T_C T_K T_G T_N T_X T_Y   
   //run_single_test_cnn(dnn_layer, R,  S,  C,  1, 1,  1,  1,  1, stonne_cfg,0); //No folding
   run_single_test_cnn(dnn_layer, R,  S,  1,  1, 1, 1,  3,  1, stonne_cfg,1); // X_=3
   run_single_test_cnn(dnn_layer, R,  S,  1,  1, 1, 1,  1,  3, stonne_cfg,2); // Y_=3
   run_single_test_cnn(dnn_layer, R,  1,  1,  1, 1, 1,  1,  3, stonne_cfg,3); 
   run_single_test_cnn(dnn_layer, R,  1,  1,  1, 1, 1,  3,  1, stonne_cfg,4);
   run_single_test_cnn(dnn_layer, 1,  S,  1,  1, 1, 1,  1,  3, stonne_cfg,5);
   run_single_test_cnn(dnn_layer, 1,  S,  1,  1, 1, 1,  3,  1, stonne_cfg,6);

  // K TEST
   run_single_test_cnn(dnn_layer, R,  S,  1,  2, 1,  1,  1,  1, stonne_cfg,7);
   run_single_test_cnn(dnn_layer, R,  S,  1,  2, 1,  1,  1,  1, stonne_cfg,8);
   std::cout << "\033[1;32mAll tests have been passed successfully\033[0m\n" << std::endl;
   delete dnn_layer;
   


}

void run_stonne_architecture_tests(layerTest layer, unsigned int num_ms) {
    unsigned int R,S,C,K,G,N,X,Y, strides; //Layer parameters
    unsigned int T_R, T_S, T_C, T_K, T_G, T_N, T_X, T_Y;
    Config stonne_cfg; 
    stonne_cfg.m_MSNetworkCfg.ms_size=num_ms;

    switch(layer) {
        case TINY:
            //R=3; S=3; C=6; K=6;N=1;X=5;Y=5; strides=1;
            R=3; S=3; C=6; K=6;G=1;N=1;X=5;Y=5; strides=1;
            T_R=3; T_S=3; T_C=1; T_K=1; T_G=1; T_N=1; T_X=3; T_Y=1;
            std::cout << "Running test TINY" << std::endl;
        break;

        case LATE_SYNTHETIC:
            R=3; S=3; C=20; K=20; G=1; N=1; X=5; Y=5; strides=1;
            T_R=3; T_S=3; T_C=1; T_K=1; T_G=1; T_N=1; T_X=3; T_Y=1;
            std::cout << "Running test LATE_SYNTHETIC" << std::endl;
        break;

        case EARLY_SYNTHETIC:
            R=3; S=3; C=6; K=6; G=1; N=1; X=20; Y=20; strides=1;
            T_R=3; T_S=3; T_C=1; T_K=1; T_G=1; T_N=1; T_X=3; T_Y=1;
            std::cout << "Running test EARLY_SYNTHETIC" << std::endl;
        break;

        case VGG_CONV11:
            R=3; S=3; C=512; K=512; G=1; N=1; X=14; Y=14; strides=1;
            T_R=3; T_S=3; T_C=1; T_K=1; T_G=1; T_N=1; T_X=3; T_Y=1;
            std::cout << "Running test VGG_CONV11" << std::endl;
        break;
 
        case VGG_CONV1:
            R=3; S=3; C=3; K=64; G=1; N=1; X=224; Y=224; strides=1;
            T_R=3; T_S=3; T_C=1; T_K=1; T_G=1; T_N=1; T_X=3; T_Y=1;
            std::cout << "Running test VGG_CONV1" << std::endl;
        break;

        default:
            std::cout << "Case not found" << std::endl;
            exit(1);
        break;
   
    }

    DNNLayer* dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G,N,X,Y,strides);
    run_single_test_cnn(dnn_layer, T_R,T_S,T_C,T_K,T_G,T_N,T_X,T_Y, stonne_cfg,0);
    std::cout << "All the tests have been passed successfully" << std::endl;
    delete dnn_layer;


    

}

void run_smart_tests() {
   std::cout << "Executing simple tests" << std::endl;
   //Hardware parameters by default. TODO modify according the new parameters added 
   Config stonne_cfg;

       //Layer parameters (See MAERI paper to find out the taxonomy meaning)
    unsigned int R=3;                                  // R
    unsigned int S=3;                                  // S
    unsigned int C=8;                                  // C
    unsigned int K=16;                                  // K
    unsigned int G=4;                                  // G
    unsigned int N=1;                                  // N
    unsigned int X=5;                                  // X
    unsigned int Y=5;                                  // Y
    unsigned int strides=1;                            // Strides

    //Tile parameters (See MAERI paper to find out the taxonomy meaning)
    unsigned int T_R=3;                                // T_R
    unsigned int T_S=3;                                // T_S
    unsigned int T_C=1;                                // T_C
    unsigned int T_K=2;                                // T_K
    unsigned int T_G=2;                                // T_G
    unsigned int T_N=1;                                // T_N
    unsigned int T_X_=1;                               // T_X
    unsigned int T_Y_=1;                               // T_Y 
   DNNLayer* dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G, N,X,Y,strides);

   //Tests varying the tile      T_R T_S T_C T_K T_G T_N T_X T_Y   
   //run_single_test_cnn(dnn_layer, R,  S,  C,  1,  1,  1,  1, stonne_cfg,0); //No folding
   
   ////////////////////////////////// TESTING GROUPS /////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Test mapping 2 G and 2 K of the same G at the same time 
   run_single_test_cnn(dnn_layer, 3,  3,  1,  2, 2, 1,  1,  1, stonne_cfg,1); // T_R=3, T_S=3, T_C=1, T_K=2, T_G=2, T_N=1, T_X_=1, T_X_=2
   //Test mapping the 4 whole K of the same group in 4 different VNs. This VNs will iterate over the different groups
   run_single_test_cnn(dnn_layer, 3,  3,  1,  4, 1, 1,  1,  1, stonne_cfg,1); // T_R=3, T_S=3, T_C=1, T_K=4, T_G=1, T_N=1, T_X_=1, T_X_=2

   delete dnn_layer;

   ////////////////////////////////// TESTING STRIDES ////////////////////////////////////////////////////////
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////

   //Tests: Trying a small filter window (2x2) with a different number of large strides which do not overlap at all
   //Test: Not overlaping and not leaving values without using (i.e., the next window is right next to the current one)
   R=2; S=2; C=1; K=1; G=1; N=1; X=18; Y=18; strides=2; 
   dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G, N,X,Y,strides);
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  1,  1, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=1
   //Test: the same but mapping 3 rows that do not overlap because of the stride
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  3,  1, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=3, T_Y_=1
   //Test: The same but mapping 3 columns
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  1,  3, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=3
   delete dnn_layer;

   //Tests: The same as before but with stride=3 which means that there is a column and row gat between two consecutives windows
    R=2; S=2; C=1; K=1; G=1; N=1; X=18; Y=18; strides=3;
   dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G, N,X,Y,strides);
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  1,  1, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=1
   //Test: the same but mapping 3 rows that do not overlap because of the stride
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  3,  1, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=3, T_Y_=1
   //Test: The same but mapping 3 columns
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  1,  3, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=3
   delete dnn_layer;
  
   //Tests the same as before but with stride=4 
   R=2; S=2; C=1; K=1; G=1; N=1; X=22; Y=22; strides=4;
   dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G, N,X,Y,strides);
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  1,  1, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=1
   //Test: the same but mapping 3 rows that do not overlap because of the stride
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  2,  1, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=2, T_Y_=1
   //Test: The same but mapping 3 columns
   run_single_test_cnn(dnn_layer, 2,  2,  1,  1, 1, 1,  1,  2, stonne_cfg,1); // T_R=2, T_S=2, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=2
   delete dnn_layer;


   

   

   //Test: Overlaping one row of input (row 4) among three different VNs with a STRIDES of 2
   stonne_cfg.m_MSNetworkCfg.ms_size=128;
   R=5; S=5; C=1; K=1; G=1; N=1; X=9; Y=9; strides=2;
   dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G, N,X,Y,strides);
   run_single_test_cnn(dnn_layer, 5,  5,  1,  1, 1, 1,  3,  1, stonne_cfg,1); // T_R=5, T_S=5, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=3, T_Y_=1
   //Test: The same but mapping the columns.
   run_single_test_cnn(dnn_layer, 5,  5,  1,  1, 1, 1,  1,  3, stonne_cfg,1); // T_R=5, T_S=5, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=3
   //Test: The same but with a larger example (increasing the input size)
   delete dnn_layer;
   R=5; S=5; C=1; K=1; G=1; N=1; X=99; Y=99; strides=2;
   dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G, N,X,Y,strides);
   run_single_test_cnn(dnn_layer, 5,  5,  1,  1, 1, 1,  3,  1, stonne_cfg,1); // T_R=5, T_S=5, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=3, T_Y_=1
   //Test: The same but mapping the columns.
   run_single_test_cnn(dnn_layer, 5,  5,  1,  1, 1, 1,  1,  3, stonne_cfg,1); // T_R=5, T_S=5, T_C=1, T_K=1, T_G=1, T_N=1, T_X_=1, T_Y_=3
   delete dnn_layer;
   stonne_cfg.m_MSNetworkCfg.ms_size=64; 


   std::cout << "\033[1;32mAll tests have been passed successfully\033[0m\n" << std::endl;



}


//This function evaluates STONNE in terms of functionallity with simple tests
void hand_tests() {
    std::cout << "Executing a hand test" << std::endl;
    unsigned int R = 3;
    unsigned int S = 3;
    unsigned int C = 256;
    unsigned int K = 384;
    unsigned int G = 1;
    unsigned int N = 1;
    unsigned int X = 15;
    unsigned int Y = 15;
    unsigned int strides = 1;
    
    //Tile variables
    unsigned int T_R=3;
    unsigned int T_S=3;
    unsigned int T_C=2;
    unsigned int T_K=2;
    unsigned int T_G=1;
    unsigned int T_N=1;
    unsigned int T_X=1;
    unsigned int T_Y=1;
  
    //Hardware parameters. TODO modify according the new parameters added 
    Config stonne_cfg;

    DNNLayer* dnn_layer = new DNNLayer(CONV, "TestLayer", R,S,C,K,G,N,X,Y,strides);
    run_single_test_cnn(dnn_layer, T_R,T_S,T_C,T_K,T_G,T_N,T_X,T_Y, stonne_cfg,0);
    std::cout << "All the tests have been passed successfully" << std::endl;
    delete dnn_layer;
    
}


