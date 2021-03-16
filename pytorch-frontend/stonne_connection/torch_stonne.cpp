#include <torch/extension.h>
#include "../../stonne/stonne_linker_src/stonne_linker.h"
#include "../../stonne/include/Config.h"

#include <iostream>
#include <tuple>

torch::Tensor simulated_linear_forward(std::string layer_name, torch::Tensor input,  torch::Tensor weight, std::string path_to_arch_file, std::string path_to_tile, float sparsity_level, bool transpose) ;



torch::Tensor simulated_matmul(std::string layer_name, torch::Tensor input,  torch::Tensor other, std::string path_to_arch_file, std::string path_to_tile, float sparsity_level) {
   /*
    * The criteria to carry out the calculation of the N-dimensional matrix multiplication is 
    * explained as follows: 
    *
    * - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    * - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    * - If the first argument is 1-dimensional and the second argument is 2-dimensional,
    *   a 1 is prepended to its dimension for the purpose of the matrix multiply. 
    *   After the matrix multiply, the prepended dimension is removed. 
    * - If the first argument is 2-dimensional and the second argument is 1-dimensional, 
    *   the matrix-vector product is returned.
    *
    * - If both arguments are at least 1-dimensional and at least one argument is 
    *   N-dimensional (where N > 2), then a batched matrix multiply is returned. 
    *   If the first argument is 1-dimensional, a 1 is prepended to its dimension 
    *   for the purpose of the batched matrix multiply and removed after. If the 
    *   second argument is 1-dimensional, a 1 is appended to its dimension for the 
    *   purpose of the batched matrix multiple and removed after. The non-matrix 
    *   (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). 
    *   For example, if input is a (j \times 1 \times n \times m)(j×1×n×m) tensor 
    *   and other is a (k \times m \times p)(k×m×p) tensor, out will be an 
    *   (j \times k \times n \times p)(j×k×n×p) tensor.
    */
    
    int first_matrix_ndim = input.dim();
    int second_matrix_ndim = other.dim();

    //If both tensors are 1-dimensional, the dot product (scalar) is returned.
    if((first_matrix_ndim == 1) &&  (second_matrix_ndim == 1)) {
        //If both tensors are 1-dimensional, the dot product (scalar) is returned.   
	torch::Tensor input_transformed = input.view({1, -1});
	torch::Tensor other_transformed = other.view({-1, 1});
	//torch::Tensor output = torch::mm(input_transformed, other_transformed);
	torch::Tensor output = simulated_linear_forward(layer_name, input_transformed, other_transformed, path_to_arch_file, path_to_tile, sparsity_level, false);
	output = output.view({output.sizes()[1]});
	return output;
    }

    //If both arguments are 2-dimensional, the matrix-matrix product is returned.
    else if((first_matrix_ndim == 2) && (second_matrix_ndim == 2)) {
        //return torch::mm(input, other); 
	return  simulated_linear_forward(layer_name, input, other, path_to_arch_file, path_to_tile, sparsity_level, false);
    }

    //If the first argument is 1-dimensional and the second argument is 2-dimensional,
    //a 1 is prepended to its dimension for the purpose of the matrix multiply. 
    //  After the matrix multiply, the prepended dimension is removed.
    else if((first_matrix_ndim == 1) &&  (second_matrix_ndim == 2)) {
        //If both tensors are 1-dimensional, the dot product (scalar) is returned.
        torch::Tensor input_transformed = input.view({1, -1});
	std::cout << "Input dimension: " << input_transformed.sizes() << std::endl;
	std::cout << "Other dimension: " << other.sizes() << std::endl;
        //torch::Tensor output = torch::mm(input_transformed, other);
	torch::Tensor output = simulated_linear_forward(layer_name, input_transformed, other, path_to_arch_file, path_to_tile, sparsity_level, false);
	std::cout << "Output dimension: " << output.sizes() << std::endl;
        output = output.view({output.sizes()[1]});
        return output;
    }
    //If the first argument is 2-dimensional and the second argument is 1-dimensional,
    //the matrix-vector product is returned.
     else if((first_matrix_ndim == 2) &&  (second_matrix_ndim == 1)) {
        //If both tensors are 1-dimensional, the dot product (scalar) is returned.
        torch::Tensor other_transformed = other.view({-1, 1});
        std::cout << "Input dimension: " << input.sizes() << std::endl;
        std::cout << "Other dimension: " << other_transformed.sizes() << std::endl;
        //torch::Tensor output = torch::mm(input, other_transformed);
	torch::Tensor output = simulated_linear_forward(layer_name, input, other_transformed, path_to_arch_file, path_to_tile, sparsity_level, false);
        std::cout << "Output dimension: " << output.sizes() << std::endl;
        output = output.view({output.sizes()[0]});
        return output;
    }

    else {
	//Adding one dimension when corresponds if the matrices are not at least 2-dimensional
	torch::Tensor input_transformed = input;
	torch::Tensor other_transformed = other;
        if((first_matrix_ndim == 1)) {
            input_transformed = input.unsqueeze(0);
	    first_matrix_ndim++;

	}

	else if (second_matrix_ndim == 1) {
            other_transformed = other.unsqueeze(1);
	    second_matrix_ndim++;

	}

	if(first_matrix_ndim != second_matrix_ndim) { //Adding extra dimensions for broadcasting
	    torch::Tensor longer_matrix = input_transformed;
	    torch::Tensor shorter_matrix = other_transformed;
	    if(second_matrix_ndim > first_matrix_ndim) { 
	        longer_matrix = other_transformed;
		shorter_matrix = input_transformed;
	    }

	    int diff = longer_matrix.dim() - shorter_matrix.dim(); 
	    for(int i=0; i<diff; i++) {
                shorter_matrix = shorter_matrix.unsqueeze(0); //Adding extra dimension to make both equal
	    }

	    input_transformed = longer_matrix;
	    other_transformed = shorter_matrix;
	    if(second_matrix_ndim > first_matrix_ndim) {
                other_transformed = longer_matrix;
                input_transformed = shorter_matrix;
            }

	}

	//At this point both matrices must equal their dimensions
	if(input_transformed.dim() != other_transformed.dim()) {
            std::cerr << "The two matrices are not broadcasted" << std::endl;
	    std::cout << "input transformed dimensions: " << input_transformed.dim() << std::endl;
	    std::cout << "other transformed dimensions: " << other_transformed.dim() << std::endl;
	    exit(1);
        }

	//We iterate over every element getting the bath sizes
	std::cout << "First matrix dimensions: " << input_transformed.sizes() << std::endl;
	std::cout << "Second matrix dimensions: " << other_transformed.sizes() << std::endl;

	//Let's check if the dimensions are broadcasted
	for(int i=0; i<input_transformed.dim()-2; i++) {
            if((input_transformed.sizes()[i] != other_transformed.sizes()[i]) && (input_transformed.sizes()[i] != 1) && (other_transformed.sizes()[i] != 1)) {
                std::cerr << "The two matrices are not broadcasted. input[" << input_transformed.sizes()[i] << "] is not compatible with other[" << other_transformed.sizes()[i] << "]" << std::endl;
		exit(1);
	    } 
	}

	//Perform the matrix multiplication
	if(input_transformed.dim() == 3) {
	       int max_dim_0 = (input_transformed.sizes()[0] > other_transformed.sizes()[0]) ? input_transformed.sizes()[0] : other_transformed.sizes()[0];
	       torch::Tensor output = torch::rand({max_dim_0,input_transformed.sizes()[1], other_transformed.sizes()[2]});
	       for(int i=0; i<max_dim_0; i++) {
	           int index_first = (input_transformed.sizes()[0] > 1) ? i : 0;
		   int index_second = (other_transformed.sizes()[0] > 1) ? i : 0;
		   std::cout << "Computing first matrix " << input_transformed[index_first].sizes() << std::endl;
		   std::cout << "Computing second matrix " << other_transformed[index_second].sizes() << std::endl; 
		   //torch::Tensor curr_output = torch::matmul(input_transformed[index_first], other_transformed[index_second]);
		   std::string layer_name_batch = layer_name + "_B_" + std::to_string(i); 
		   torch::Tensor curr_output = simulated_linear_forward(layer_name_batch, input_transformed[index_first], other_transformed[index_second], path_to_arch_file, path_to_tile, sparsity_level, false);
		   output.slice(0,i,i+1)=curr_output;
	       }

	       return output;

	}

	else if(input_transformed.dim() == 4) {
            std::cout << "Mtrix with dimensions 4" << std::endl;
	    int max_dim_0 = (input_transformed.sizes()[0] > other_transformed.sizes()[0]) ? input_transformed.sizes()[0] : other_transformed.sizes()[0];
	    int max_dim_1 = (input_transformed.sizes()[1] > other_transformed.sizes()[1]) ? input_transformed.sizes()[1] : other_transformed.sizes()[1];
            torch::Tensor output = torch::rand({max_dim_0,max_dim_1, input_transformed.sizes()[2], other_transformed.sizes()[3]});
               for(int i=0; i<max_dim_0; i++) {
                   int index_first_0 = (input_transformed.sizes()[0] > 1) ? i : 0;
                   int index_second_0 = (other_transformed.sizes()[0] > 1) ? i : 0;
		   for(int j=0; j<max_dim_1; j++) {
                       int index_first_1 = (input_transformed.sizes()[1] > 1) ? j : 0;
                       int index_second_1 = (other_transformed.sizes()[1] > 1) ? j : 0;
                       //torch::Tensor curr_output = torch::matmul(input_transformed[index_first_0][index_first_1], other_transformed[index_second_0][index_second_1]);
		       std::string layer_name_batch=layer_name+"_B_"+std::to_string(i)+"_"+std::to_string(j); 
		       torch::Tensor curr_output = simulated_linear_forward(layer_name_batch, input_transformed[index_first_0][index_first_1], other_transformed[index_second_0][index_second_1], path_to_arch_file, path_to_tile, sparsity_level, false);
                       output.slice(0,i,i+1).slice(1,j,j+1)=curr_output;
		   }
               }

               return output;

	}

	else {
            std::cerr << ">5-dimension matrix multiplications not supported" << std::endl;
	    exit(1);
        
	}

    }

    
}
// Types must match:
//    stride: Tuple[int, ...]
//    padding: Tuple[int, ...]
//    dilation: Tuple[int, ...]
//    transposed: bool
//    output_padding: Tuple[int, ...]
//    groups: int
//    padding_mode: str
//    weight: Tensor
//    bias: Optional[Tensor]
//conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) 
torch::Tensor simulated_conv_forward(std::string layer_name, torch::Tensor input,  torch::Tensor weight,c10::ArrayRef<long int> stride, c10::ArrayRef<long int> padding, c10::ArrayRef<long int> dilation, int64_t groups, std::string path_to_arch_file, std::string path_to_tile, float sparsity_level) {
    //Here starts the function
    //Creating config file to find out if we are going to run a dense or sparse simulation
    Config stonne_cfg; 
    if(path_to_arch_file != "") {
        stonne_cfg.loadFile(path_to_arch_file);
    }

    if(stonne_cfg.sparsitySupportEnabled()) {
        std::cout << "Sparsity support enabled with ratio" << sparsity_level <<  std::endl;
	//im2col to the inputs
	int R = weight.sizes()[2];
        int S = weight.sizes()[3];
        int C = input.sizes()[1]; //All the channels. Note this could not be the same in weight.sizes[1] (i.e., filter channels) as the groups could reduce these last ones.
	//In this case, we send the complete number of input channels, and the callee
	//will have to be aware of this and run C/G if  groups exist.
        int K = weight.sizes()[0];
        int G = (int) groups;
        int N = input.sizes()[0];
        int X = input.sizes()[2]; //Add padding sizes?
        int Y = input.sizes()[3];
	namespace F = torch::nn::functional;
	torch::Tensor input_im2col = F::unfold(input, F::UnfoldFuncOptions({R, S}).padding(padding).stride(stride).dilation(dilation)); //This function returns a 3D tensor [N, R*S*C, number_of_outputs]
	//Getting raw data
        float* KN_input_raw = (float*) input_im2col.data_ptr();
        float* MK_weight_raw = (float*) weight.data_ptr();
	//Creating output tensor
        int H_out = ((X + 2*padding[0] - dilation[0] * (R-1) - 1) / stride[0]) + 1;
        int W_out = ((Y + 2*padding[1] - dilation[1] * (S-1) - 1) / stride[0]) + 1;
        torch::Tensor output = torch::rand({N,K,H_out, W_out});
        float* output_raw = (float*) output.data_ptr();
  
	//Note that since STONNE only supports sparse GEMM operation, we have to turn
	// the input to im2col format and run a GEMM operation instead a CONVOLUTION
	//Getting GEMM dimensions
	//MK matrix are the weights
        int gemm_M = K; //Number of filters (weight.sizes()[0];) (i.e., rows MK)
	int gemm_K = input_im2col.sizes()[1]; //window size (i.e., columns MK)
	int gemm_N = input_im2col.sizes()[2]; //0 is batch dim, 1 is K

        simulateSparseGemmForward(layer_name, KN_input_raw, MK_weight_raw, output_raw, N, G, gemm_M, gemm_K, gemm_N, sparsity_level, stonne_cfg, MK_STA_KN_STR); //Keeping MK stationary as they are the weights
        return output;


    }
    else if(!stonne_cfg.convOperationSupported()) { //IF CONV itself is not supported, we run it as a GEMM (e.g., the TPU)
        //im2col to the inputs
        int R = weight.sizes()[2];
        int S = weight.sizes()[3];
        int C = input.sizes()[1]; //All the channels. Note this could not be the same in weight.sizes[1] (i.e., filter channels) as the groups could reduce these last ones.
        //In this case, we send the complete number of input channels, and the callee
        //will have to be aware of this and run C/G if  groups exist.
        int K = weight.sizes()[0];
        int G = (int) groups;
        int N = input.sizes()[0];
        int X = input.sizes()[2]; //Add padding sizes?
        int Y = input.sizes()[3];
        namespace F = torch::nn::functional;
        torch::Tensor input_im2col = F::unfold(input, F::UnfoldFuncOptions({R, S}).padding(padding).stride(stride).dilation(dilation)); //This function returns a 3D tensor [N, R*S*C, number_of_outputs]
        //Getting raw data
        //float* KN_input_raw = (float*) input_im2col.data_ptr();
	torch::Tensor KN_input_transposed=input_im2col;
        KN_input_transposed = input_im2col[0].transpose(1, 0);
        KN_input_transposed = KN_input_transposed.contiguous(); //Contigous is to transform the container so that it stores the data by the transpose
	float* KN_input_raw = (float*) KN_input_transposed.data_ptr();


        float* MK_weight_raw = (float*) weight.data_ptr();
        //Creating output tensor
        int H_out = ((X + 2*padding[0] - dilation[0] * (R-1) - 1) / stride[0]) + 1;
        int W_out = ((Y + 2*padding[1] - dilation[1] * (S-1) - 1) / stride[0]) + 1;
        torch::Tensor output = torch::rand({N,K,H_out, W_out});
        float* output_raw = (float*) output.data_ptr();

        //Note that since STONNE only supports sparse GEMM operation, we have to turn
        // the input to im2col format and run a GEMM operation instead a CONVOLUTION
        //Getting GEMM dimensions
        //MK matrix are the weights
        int gemm_M = K; //Number of filters (weight.sizes()[0];) (i.e., rows MK)
        int gemm_K = input_im2col.sizes()[1]; //window size (i.e., columns MK)
        int gemm_N = input_im2col.sizes()[2]; //0 is batch dim, 1 is K
        simulateDenseGemmForward(layer_name, KN_input_raw, MK_weight_raw, output_raw, N, G, gemm_M, gemm_K, gemm_N, path_to_tile, stonne_cfg);
  //      simulateSparseGemmForward(layer_name, KN_input_raw, MK_weight_raw, output_raw, N, G, gemm_M, gemm_K, gemm_N, sparsity_level, stonne_cfg, MK_STA_KN_STR); //Keeping MK stationary as they are the weights
        return output;

    }

    else {
        //Dense piece of code
        //Getting conv layer parameters
        int R = weight.sizes()[2];
        int S = weight.sizes()[3];
        int C = input.sizes()[1]; //All the channels. Note this could not be the same in weight.sizes[1] (i.e., filter channels) as the groups could reduce these last ones.
        //In this case, we send the complete number of input channels, and the callee
        //will have to be aware of this and run C/G if  groups exist.;
        int K = weight.sizes()[0];
        int G = (int) groups;
        int N = input.sizes()[0];
        int X = input.sizes()[2]; //Add padding sizes?
        int Y = input.sizes()[3]; //Add padding sizes?
        int strides = stride[0];
	int pad_x = padding[0];
	int pad_y = padding[1];

        //Getting raw data 
        float* input_raw = (float*) input.data_ptr();
        float* weight_raw = (float*) weight.data_ptr();

        //Creating output tensor 
        int H_out = ((X + 2*padding[0] - dilation[0] * (R-1) - 1) / stride[0]) + 1; 
        int W_out = ((Y + 2*padding[1] - dilation[1] * (S-1) - 1) / stride[0]) + 1;
        torch::Tensor output = torch::rand({N,K,H_out, W_out});
        float* output_raw = (float*) output.data_ptr();

        simulateDenseConvForward(layer_name, input_raw, weight_raw, output_raw, R, S, C, K, G, N, X, Y, H_out, W_out, strides, pad_x, pad_y, path_to_tile, stonne_cfg);
        return output;
    }
}



torch::Tensor simulated_linear_forward(std::string layer_name, torch::Tensor input,  torch::Tensor weight, std::string path_to_arch_file, std::string path_to_tile, float sparsity_level, bool weights_organized_by_rows) {
    //Here starts the function
    //Creating config file to find out if we are going to run a dense or sparse simulation
    Config stonne_cfg; 
    if(path_to_arch_file != "") {
        stonne_cfg.loadFile(path_to_arch_file);
    }

     torch::Tensor weight_changed=weight;
     weight_changed = weight.transpose(1, 0);
     weight_changed = weight_changed.contiguous(); //Contigous is to transform the container so that it stores the data by the transpose

    //Getting the data
    int input_n_dim = input.dim();
    int M = input.sizes()[input_n_dim-2]; // Batch size. The dimension previous to the last one
    int K = input.sizes()[input_n_dim-1]; //Number of input neurons. Last dimension
    int N;
    if(weights_organized_by_rows) {
        N = weight.sizes()[0]; //Number of output neurons

    }
    else {
        N = weight.sizes()[1];
    }
    float* MK_input_raw = (float*) input.data_ptr();
    float* KN_weight_raw ;
	//Creating output tensor
    torch::Tensor output;
    if(input_n_dim == 2) {
        output = torch::rand({M, N}); //M is batch size and N is the number of output neurons
    }

    else {
        output = torch::rand({1, M, N});
    }
    float* output_raw = (float*) output.data_ptr();
    int gemm_M = M; 
    int gemm_K = K;
    int gemm_N = N;

    if(stonne_cfg.sparsitySupportEnabled()) {
      if(weights_organized_by_rows) { //Sparse accepts matrix KN organized as KN
         KN_weight_raw = (float*) weight_changed.data_ptr();
      }
      else {
         KN_weight_raw = (float*) weight.data_ptr(); //No tranpose
      }
      std::cout << "Calling to the simulation with gemm_M=" << gemm_M << " gemm_N=" << gemm_N << std::endl;
      simulateSparseGemmForward(layer_name, KN_weight_raw, MK_input_raw, output_raw, 1, 1, gemm_M, gemm_K, gemm_N, sparsity_level, stonne_cfg, MK_STR_KN_STA);
    }

    else {
	if(weights_organized_by_rows) {
            KN_weight_raw = (float*) weight.data_ptr();
        }

	else {
            KN_weight_raw = (float*) weight_changed.data_ptr();
        }
        simulateDenseGemmForward(layer_name, KN_weight_raw, MK_input_raw, output_raw,1, 1, gemm_M, gemm_K, gemm_N, path_to_tile, stonne_cfg);
	std::cout << "The value of M is " << gemm_M << std::endl;
	std::cout << "The value of N is " << gemm_N << std::endl;
    }

    return output;

   
}



torch::Tensor simulated_conv_backwards(torch::Tensor z, std::string mi_string) {
    std::cout << "Running a simulated layer backward method" << std::endl;
    std::cout << mi_string << std::endl; 
    std::cout << z.sizes() << std::endl;
    namespace F = torch::nn::functional;
    torch::Tensor new_tensor = F::unfold(z, F::UnfoldFuncOptions({2, 2}).padding(0).stride(1));
 //   float* data = (float*) new_tensor.data_ptr();
//    for(int i=0; i<20; i++) {
//        data[i]=23.0;
//    }
    return new_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("simulated_conv_forward", &simulated_conv_forward, "Simulated convolution forward");
  m.def("simulated_conv_backward", &simulated_conv_backwards, "Simulated convolution backward");
  m.def("simulated_linear_forward", &simulated_linear_forward, "Simulated linear forward");
  m.def("simulated_matmul", &simulated_matmul, "Simulated Matrix multiplication");
}
