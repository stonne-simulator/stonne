#include <iostream>
#include "stonne_linker.h"
#include <unistd.h>
#include "STONNEModel.h"
#include "Tile.h"
#include "Config.h"
#include "types.h"
#include "testbench.h"
#include "utility.h"
#include <string>

float* Transform_Ifmap_Memory (const float* bottom_data, const int C, const int X, const int Y, const int pad_x, const int pad_y);

float* Transform_Filters_Memory (const float* weights, const int K, const int G, const int C, const int R, const int S);

void Transform_Ofmap_Memory (const float* ofmap_data, float* top_data, const int K, const int X_, const int Y_);

float* Transform_Ifmap_Memory (const float* bottom_data, const int C, const int X, const int Y, const int pad_x, const int pad_y) {
    const int n_channels=C;
    const int input_y=X;
    const int input_x=Y;

    const int input_y_pad=input_y + 2*pad_y;
    const int input_x_pad=input_x + 2*pad_x;
    int size_channel=input_y*input_x;
    int n=n_channels*(input_y_pad*input_x_pad);
    
    float* data_to_send = new float[n]; //Creating piece of memory that will use the simulator
    //Adding y padding

    for(int i=0; i<n; i++) {
	data_to_send[i]=0.0;
    }
    for(int i=0; i<n_channels; i++) {
        for(int y=0; y<input_y; y++) {
            for(int x=0; x<input_x; x++) {
                data_to_send[(n_channels*((y+pad_y)*input_x_pad+x+pad_x)) + i]=bottom_data[i*size_channel + y*(input_x) + x];
            }
        }
    }	

  
    return data_to_send;

}

float* Transform_Filters_Memory (const float* weights, const int K, const int G, const int C, const int R, const int S) {
    
    const int n_channels=C / G;
    const int kernel_y=R;
    const int kernel_x=S;
    const int n_filters=K;    //this->num_output_;
    int size_channel=kernel_y*kernel_x;
    int size_filter=size_channel*n_channels;
    int n=size_filter*n_filters;

    float* filters_to_send = new float[n]; //Creating piece of memory that will use the simulator
    for(int n_f=0; n_f < n_filters; n_f++) {
        for(int i=0; i<n_channels; i++) {
            for(int y=0; y<kernel_y; y++) {
                for(int x=0; x<kernel_x; x++) {
                    filters_to_send[n_f*size_filter+(n_channels*(y*kernel_x+x)) + i]=weights[n_f*size_filter+i*size_channel + y*kernel_x + x];
                }
            }
        }
    }


    return filters_to_send;

}


void Transform_Ofmap_Memory (const float* ofmap_data, float* top_data, const int K, const int X_, const int Y_) {
    const int n_channels=K; //n_filters
    const int output_y=X_;
    const int output_x=Y_;

    int size_channel=output_y*output_x;
    int n=n_channels*size_channel;
    for(int i=0; i<n_channels; i++) {
        for(int y=0; y<output_y; y++) {
            for(int x=0; x<output_x; x++) {
                //data_to_send[(n_channels*(y*input_x+x)) + i]=bottom_data[i*size_channel + y*input_x + x];
                top_data[i*size_channel+y*output_x+x]=ofmap_data[(n_channels*(y*output_x+x)) + i]; //Filling top_data
            }
        }
    }


}


void simulateDenseConvForward(std::string layer_name, float* input, float* weight, float* output, int R, int S, int C, int K, int G, int N, int X, int Y, int X_, int Y_, int strides, int pad_x, int pad_y, std::string path_to_tile, Config stonne_cfg) {
  //Modifying layer name to avoid / characters
  /* const string fixed_layer_name= this->layer_param_.name();
   string layer_name="";
   for(int i=0; i<fixed_layer_name.length(); i++) {
       if (fixed_layer_name[i]=='/') {
           layer_name+="_"; // _ character is changed by /
       }

      else {
          layer_name+=fixed_layer_name[i];
      }
   }
*/
   //Updating X and Y with pad values
   //const int pad_y=this->pad_.cpu_data()[0]; //alto
   const int ifmap_size=C*((X+2*pad_x)*(Y+2*pad_y));
   const int ofmap_size = K*X_*Y_; //X_ and Y_ include padding
   std::cout << "Executing layer " << layer_name << std::endl;
   if(path_to_tile == "") {
	   std::cout << "Tile file parameters must be specified" << std::endl;
	   exit(1);
   }

   //Loading the tile
   Tile tile(path_to_tile);


   float* ifmap_to_send=Transform_Ifmap_Memory(input, C, X, Y, pad_x, pad_y) ;
   float* filters_to_send=Transform_Filters_Memory(weight, K, G, C, R, S);
   float* ofmap_raw = new float[ofmap_size];


   //Tile parameters
   unsigned int T_R = tile.get_T_R();
   unsigned int T_S = tile.get_T_S();
   unsigned int T_C = tile.get_T_C();
   unsigned int T_K = tile.get_T_K();
   unsigned int T_G = tile.get_T_G();
   unsigned int T_N = tile.get_T_N();
   unsigned int T_X_ = tile.get_T_X_();
   unsigned int T_Y_ = tile.get_T_Y_();


   //Executing the accelerator
   Stonne* stonne_instance = new Stonne(stonne_cfg);
   stonne_instance->loadDNNLayer(CONV, layer_name, R, S, C, K, G, N, X+2*pad_x, Y+2*pad_y, strides, (address_t) ifmap_to_send, (address_t)filters_to_send, (address_t)ofmap_raw, CNN_DATAFLOW);
   stonne_instance->loadTile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_);
   stonne_instance->run(); //Running the accelerator and generates the output in ofmap_raw
   //sequential_layer(R, S, C, K, G, N, X, Y, strides, (address_t)ifmap_to_send, (address_t)filters_to_send, (address_t)ofmap_raw);

   Transform_Ofmap_Memory(ofmap_raw, output, K, X_, Y_); // Transform simulator memory format to caffe format.     

   //Deleting objects
   delete[] ofmap_raw;
   delete[] ifmap_to_send;
   delete[] filters_to_send;
   delete stonne_instance;

}

void simulateSparseGemmForward(std::string layer_name, float* KN_matrix_raw, float* MK_matrix_raw, float* output_raw, int N, int G, int gemm_M, int gemm_K, int gemm_N, float sparsity_level, Config stonne_cfg, Dataflow dataflow) { 

    //We have to modify the parameters based on the groups
    int gemm_M_grouped = gemm_M / G; 
    int gemm_K_grouped = gemm_K / G;
    int gemm_N_grouped = gemm_N;

    int weight_offset_ = gemm_M_grouped * gemm_K_grouped;
    int col_offset_ = gemm_K_grouped * gemm_N_grouped;
    int out_offset_ = gemm_M_grouped * gemm_N_grouped;

    for(int n=0; n<N; n++) {
	std::string layer_name_batch=layer_name+"_batch_"+std::to_string(n);
        for(int g=0; g<G; g++) {  //Every group is calculated with a different GEMM
		std::string layer_name_group = (layer_name_batch+("_group_"+std::to_string(g)));
             float* pointer_MK_dense_matrix = (float*) MK_matrix_raw  + weight_offset_ * g;
             float* pointer_KN_dense_matrix = (float*) KN_matrix_raw + col_offset_ * g;

             //Setting sparsity in weights
	     float* MK_dense_matrix;
	     float* KN_dense_matrix;
	     if(dataflow == MK_STA_KN_STR) {
		 KN_dense_matrix = new float[gemm_K_grouped*gemm_N_grouped];
                 MK_dense_matrix = generatePrunnedMatrix(pointer_MK_dense_matrix, gemm_M_grouped*gemm_K_grouped, sparsity_level);

	         for(int i=0; i<gemm_N_grouped*gemm_K_grouped; i++) {
                     KN_dense_matrix[i]=pointer_KN_dense_matrix[i];
                }

	    }

	    else {
		MK_dense_matrix = new float[gemm_M_grouped*gemm_K_grouped];
                KN_dense_matrix = generatePrunnedMatrix(pointer_KN_dense_matrix, gemm_K_grouped*gemm_N_grouped, sparsity_level);

                 for(int i=0; i<gemm_M_grouped*gemm_K_grouped; i++) {
                     MK_dense_matrix[i]=pointer_MK_dense_matrix[i];
                }

	    }


             float* acc_output = (float*)output_raw + out_offset_ * g;

	     unsigned int* acc_bitmap = new unsigned int[gemm_M_grouped*gemm_N_grouped]; //Currently is not generated by the accelerator


             //Generating bitmaps
             unsigned int* MK_bitmap = generateBitMapFromDense(MK_dense_matrix, gemm_M_grouped, gemm_K_grouped, GEN_BY_ROWS);
             unsigned int* KN_bitmap = generateBitMapFromDense(KN_dense_matrix, gemm_K_grouped, gemm_N_grouped, GEN_BY_COLS);

	    
             //Generating sparse matrix
             float* MK_sparse_matrix = generateMatrixSparseFromDense(MK_dense_matrix, MK_bitmap, gemm_M_grouped, gemm_K_grouped, GEN_BY_ROWS);
             float* KN_sparse_matrix = generateMatrixSparseFromDense(KN_dense_matrix, KN_bitmap, gemm_K_grouped, gemm_N_grouped, GEN_BY_COLS);

	     //Running STONNE
            Stonne* stonne_instance = new Stonne(stonne_cfg);

            stonne_instance->loadGEMM(layer_name_group, gemm_N_grouped, gemm_K_grouped, gemm_M_grouped, MK_sparse_matrix, KN_sparse_matrix, MK_bitmap, KN_bitmap, acc_output, acc_bitmap, dataflow ); //Loading GEMM
            stonne_instance->run(); //Running the simulator
            delete[] MK_bitmap;
            delete[] KN_bitmap;
            delete[] MK_dense_matrix;
            delete[] KN_dense_matrix;
            delete[] MK_sparse_matrix;
            delete[] KN_sparse_matrix;
            delete[] acc_bitmap;
            delete stonne_instance;

	}
    }
}

void simulateSparseGemmWithBitmapsForward(std::string layer_name, float* KN_matrix_raw, float* MK_matrix_raw, float* output_raw, int N, int G, int gemm_M, int gemm_K, int gemm_N, unsigned int* MK_bitmap, unsigned int* KN_bitmap, Config stonne_cfg, Dataflow dataflow) { 

    //We have to modify the parameters based on the groups
    int gemm_M_grouped = gemm_M / G; 
    int gemm_K_grouped = gemm_K / G;
    int gemm_N_grouped = gemm_N;

    int weight_offset_ = gemm_M_grouped * gemm_K_grouped;
    int col_offset_ = gemm_K_grouped * gemm_N_grouped;
    int out_offset_ = gemm_M_grouped * gemm_N_grouped;

    for(int n=0; n<N; n++) {
	std::string layer_name_batch=layer_name+"_batch_"+std::to_string(n);
        for(int g=0; g<G; g++) {  //Every group is calculated with a different GEMM
		std::string layer_name_group = (layer_name_batch+("_group_"+std::to_string(g)));
             float* MK_sparse_matrix = (float*) MK_matrix_raw  + weight_offset_ * g;
             float* KN_sparse_matrix = (float*) KN_matrix_raw + col_offset_ * g;

	     unsigned int* pointer_MK_bitmap = (unsigned int*) MK_bitmap + weight_offset_ * g;
	     unsigned int* pointer_KN_bitmap = (unsigned int*) KN_bitmap + col_offset_ * g;


             float* acc_output = (float*)output_raw + out_offset_ * g;

	     unsigned int* acc_bitmap = new unsigned int[gemm_M_grouped*gemm_N_grouped]; //Currently is not generated by the accelerator
	    
	    //Running STONNE
            Stonne* stonne_instance = new Stonne(stonne_cfg);

            stonne_instance->loadGEMM(layer_name_group, gemm_N_grouped, gemm_K_grouped, gemm_M_grouped, MK_sparse_matrix, KN_sparse_matrix, pointer_MK_bitmap, pointer_KN_bitmap, acc_output, acc_bitmap, dataflow ); //Loading GEMM
            stonne_instance->run(); //Running the simulator
            delete[] acc_bitmap;
            delete stonne_instance;

	}
    }
}



void simulateDenseGemmForward(std::string layer_name, float* KN_matrix_raw, float* MK_matrix_raw, float* output_raw, int N, int G, int gemm_M, int gemm_K, int gemm_N, std::string path_to_tile, Config stonne_cfg) {
   if(path_to_tile == "") {
       std::cout << "Tile file parameters must be specified" << std::endl;
       exit(1);
   }

   //Loading the tile
   Tile tile(path_to_tile);


   int gemm_M_grouped = gemm_M / G;
   int gemm_K_grouped = gemm_K / G;
   int gemm_N_grouped = gemm_N;

   int weight_offset_ = gemm_M_grouped * gemm_K_grouped;
   int col_offset_ = gemm_K_grouped;
   int out_offset_ = gemm_M_grouped * gemm_N_grouped;
   float* sub_KN_dense_matrix = new float[gemm_N_grouped*gemm_K_grouped];

   //Setting accelerator parameters. Mapping GEMM into a conv tile and layer
   //int S = gemm_K_grouped;
   //int K = gemm_N_grouped;
   //int N = gemm_M_grouped;
   

   //Tile parameters
   int T_K = tile.get_T_S();
   int T_N = tile.get_T_K();
   int T_M = tile.get_T_N();


   for(int n=0; n<N; n++) {
       std::string layer_name_batch=layer_name+"_batch_"+std::to_string(n);
       for(int g=0; g<G; g++) {  //Every group is calculated with a different GEMM
           std::string layer_name_group = (layer_name_batch+("_group_"+std::to_string(g)));
	   float* pointer_MK_dense_matrix = (float*) MK_matrix_raw  + weight_offset_ * g;
           float* pointer_KN_dense_matrix = (float*) KN_matrix_raw + col_offset_ * g;
	   //Since the offset in this case is per column, we have to use a submatrix. 
	   for(int i=0; i<gemm_N_grouped; i++) {
               for(int j=0; j<gemm_K_grouped; j++) {
		   float* sub_pointer=(float*) KN_matrix_raw + col_offset_ * g;
                   sub_KN_dense_matrix[i*gemm_K_grouped+j]=sub_pointer[i*gemm_K+j];
	       }
           }

	   pointer_KN_dense_matrix = sub_KN_dense_matrix;
	   float* acc_output = (float*)output_raw + out_offset_ * g;


           Stonne* stonne_instance = new Stonne(stonne_cfg); //Creating the instance of the simulator
           stonne_instance->loadFCLayer(layer_name_group, gemm_N_grouped, gemm_K_grouped, gemm_M_grouped, (address_t) pointer_MK_dense_matrix, (address_t) pointer_KN_dense_matrix, (address_t) acc_output);
           stonne_instance->loadFCTile(T_K, T_M, T_N);
           stonne_instance->run(); //Running the accelerator and generates the output in ofmap_raw

           delete stonne_instance;
       }
   }     
   delete[] sub_KN_dense_matrix;

}

