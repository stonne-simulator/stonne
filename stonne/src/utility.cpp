#include <math.h>
#include "utility.h"
#include <assert.h>
#include <vector>
#include <algorithm>
bool isNum(std::string str) {
  std::istringstream sin(str);
  double d;
  char c;
  if(!(sin >> d)) {
    return false;
  }
  else if(sin >> c) {
    return false;
  }
  else {
    return true;
  }
}

std::string getstr(std::istringstream& instr) {
  std::string str;
  while(instr >> str) {
    if(isNum(str)) {
      return str;
    }
    else {
      continue;
    }
  }
}

bool ispowerof2(unsigned int x) {
    return x && !(x & (x - 1));
}

std::string get_string_adder_configuration(adderconfig_t config) {
    switch(config) {
        case ADD_2_1:
        return "ADD_2_1";
        break;

        case ADD_3_1:
        return "ADD_3_1";
        break;

        case ADD_1_1_PLUS_FW_1_1:
        return "ADD_1_1_PLUS_FW_1_1";
        break;

        case FW_2_2:
        return "FW_2_2";
        break;

        case NO_MODE:
        return "NO_MODE";
        break;

        case FOLD:
        return "FOLD";
        break;

        default:
            assert(false);
    }
}

std::string get_string_fwlink_direction(fl_t fl_direction) {
    switch(fl_direction) {
        case SEND:
            return "SEND";
            break;

        case RECEIVE:
            return "RECEIVE";
            break;

        case NOT_CONFIGURED:
            return "NOT_CONFIGURED";
            break;

        default: 
             assert(false);
    }

 
}

std::string get_string_reduce_network_type(ReduceNetwork_t reduce_network_type) {
    switch(reduce_network_type) {
        case ASNETWORK: 
            return "ASNETWORK";
            break;
        case FENETWORK:
            return "FENETWORK";
            break;
	case TEMPORALRN:
	    return "TEMPORALRN";
	    break;
        default:
            assert(false);
            break;
    }
}

ReduceNetwork_t get_type_reduce_network_type(std::string reduce_network_type) {
        if(reduce_network_type=="ASNETWORK") {
            return ASNETWORK;
        }
        else if(reduce_network_type=="FENETWORK") {
            return FENETWORK;
        }

	else if(reduce_network_type=="TEMPORALRN") {
            return TEMPORALRN;
	}
        else {
            std::cout << reduce_network_type << " Not found" << std::endl;
            assert(false);
        }
 
}

std::string get_string_multiplier_network_type(MultiplierNetwork_t multiplier_network_type) {
    switch(multiplier_network_type) {
        case LINEAR: 
            return "LINEAR";
            break;
        case OS_MESH:
            return "OS_MESH";
            break;
        default:
            assert(false);
            break;
    }
}

MultiplierNetwork_t get_type_multiplier_network_type(std::string multiplier_network_type) {
        if(multiplier_network_type=="LINEAR") {
            return LINEAR;
        }
        else if(multiplier_network_type=="OS_MESH") {
            return OS_MESH;
        }

        else {
            std::cout << multiplier_network_type << " Not found" << std::endl;
            assert(false);
        }
 
}

std::string get_string_memory_controller_type(MemoryController_t memory_controller_type) {
    switch(memory_controller_type) {
        case MAERI_DENSE_WORKLOAD:
            return "MAERI_DENSE_WORKLOAD";
            break;
        case SIGMA_SPARSE_GEMM:
            return "SIGMA_SPARSE_GEMM";
            break;
            /////
	
	case TPU_OS_DENSE:
	    return "TPU_OS_DENSE";
	    break;

	case SNAPEA_OS_DENSE:
	    return "SNAPEA_OS_DENSE";
        default:
            assert(false);
            break;
    }
}

MemoryController_t get_type_memory_controller_type(std::string memory_controller_type) {
        if(memory_controller_type=="MAERI_DENSE_WORKLOAD") {
            return MAERI_DENSE_WORKLOAD;
        }
        else if(memory_controller_type=="SIGMA_SPARSE_GEMM") {
            return SIGMA_SPARSE_GEMM;
        }
        /////

	else if(memory_controller_type=="TPU_OS_DENSE") {
            return TPU_OS_DENSE;
        }

	else if(memory_controller_type == "SNAPEA_OS_DENSE") {
            return SNAPEA_OS_DENSE;
	}
        else {
            std::cout << memory_controller_type << " Not found" << std::endl;
            assert(false);
        }

}

Dataflow get_type_dataflow_type(std::string dataflow_type) {
        if(dataflow_type=="CNN_DATAFLOW") {
            return CNN_DATAFLOW;
        }
        else if(dataflow_type=="MK_STA_KN_STR") {
            return MK_STA_KN_STR;
        }

	else if(dataflow_type=="MK_STR_KN_STA") {
            return MK_STR_KN_STA;
	}
	/////
	else if(dataflow_type=="SPARSE_DENSE_DATAFLOW") {
	    return SPARSE_DENSE_DATAFLOW;
	}
	
        else {
            std::cout << dataflow_type << " Not found" << std::endl;
            assert(false);
        }

}

std::string get_string_dataflow_type(Dataflow dataflow) {
    switch(dataflow) {
        case CNN_DATAFLOW:
            return "CNN_DATAFLOW";
            break;
        case MK_STR_KN_STA:
            return "MK_STR_KN_STA";
            break;

	case MK_STA_KN_STR:
	    return "MK_STA_KN_STR";
	    break;
	/////
	case SPARSE_DENSE_DATAFLOW:
	    return "SPARSE_DENSE_DATAFLOW";
	    break;
        default:
            assert(false);
            break;
    }
}












std::string ind(unsigned int indent) {
    std::string str="";
    for(int i=0; i<indent; i++) {
        str+=" ";
    }

   return str;
}

float* generatePrunnedMatrix(const float* src_matrix, unsigned int size, float pr_ratio) {
    float * dst_matrix = new float[size];
    for(int i=0; i<size; i++) {
        dst_matrix[i]=fabs(src_matrix[i]);
    }
    int n = size * pr_ratio;
    std::cout << "n to crib is " << n << std::endl;
    float* begin = dst_matrix;
    float* end = begin + size;
    float* nth = begin + n;
    std::nth_element(begin, nth, end); //Sorting 
    float pivot = dst_matrix[n];
    //Copying and prunning
    for(int i=0; i<size; i++) {
        float value_abs = fabs(src_matrix[i]); 
	if(value_abs < pivot) {
            dst_matrix[i]=0.0; //Prunned
	}
	else {
            dst_matrix[i]=src_matrix[i]; //Not prunned
        }
    }

    return dst_matrix;

}
float* generateMatrixDense(unsigned int rows, unsigned int cols, unsigned int sparsity) {
    float* matrix = new float[rows*cols];
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
	    float value;
            unsigned int sparse_yes=rand()%100;
	    if(sparse_yes <= sparsity) {
                value=0.0;
	    }

	    else {
                value=rand()%10 + 1;
	    }

	    matrix[i*cols+j]=value;

	}
    }

    return matrix;
}

unsigned int* generateBitMapFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type) {
    unsigned int* bitMap = new unsigned int[rows*cols];
    if(gen_type==GEN_BY_ROWS)  {
	unsigned int non_zeros = 0;
        for(int i=0; i<rows; i++) {
	    non_zeros=0;
            for(int j=0; j<cols; j++) {
                if(denseMatrix[i*cols+j] != 0.0) { 
                    bitMap[i*cols+j]=1;
		    non_zeros++;
	        }

	        else {
                    bitMap[i*cols+j]=0;
	        }


	    }

	    if(non_zeros<3) {
                for(int j=0; (j<3)&&(non_zeros<3); j++) {
		    if(bitMap[i*cols+j] == 0) {
                        bitMap[i*cols+j]=1; //Even if this is a 0. This is to make sure sizes greater than 3
			non_zeros++;
		    }
                }
            }

        }
    }

    else if(gen_type==GEN_BY_COLS) {
    	unsigned int non_zeros = 0;
        for(int j=0; j<cols; j++) {
	    non_zeros=0;
            for(int i=0; i<rows; i++) {
                if(denseMatrix[i*cols+j] != 0.0) { //TODO Warning with the accuracy
                    bitMap[i*cols+j]=1;
		    non_zeros++;
	        }

	        else {
                    bitMap[i*cols+j]=0;
	        }


	    }

	    if(non_zeros<3) {
                for(int i=0; (i<3)&&(non_zeros<3); i++) {
		    if(bitMap[i*cols+j]==0) {
                         bitMap[i*cols+j]=1; //Even if this is a 0. This is to make sure sizes greater than 3
			 non_zeros++;
		    }
                }
            }

        }
    }

    return bitMap;

}	

/////
float* generateMatrixSparseFromDenseNoBitmap(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type) {
	std::vector<float> elements;
        if(gen_type==GEN_BY_ROWS) {
            for(int i=0; i<rows; i++) {
                int non_zeros=0;
                for(int j=0; j<cols; j++) {
                    if(denseMatrix[i*cols+j]!=0.0) {
                        elements.push_back(denseMatrix[i*cols+j]);
                        non_zeros++;
		    }
                }
                if(non_zeros==0)
                	elements.push_back(0.0);
	    }
	}

        else { //In columns order (KN)
            for(int j=0; j<cols; j++) {
            int non_zeros=0;
                for(int i=0; i<rows; i++) {
                    if(denseMatrix[i*cols+j]!=0.0) {
                        elements.push_back(denseMatrix[i*cols+j]);
                        non_zeros++;
		    }
		}
		if(non_zeros==0)
                	elements.push_back(0.0);
            }
	}	

	float* sparseMatrix = new float[elements.size()];
	for(int i=0; i<elements.size(); i++) {
            sparseMatrix[i]=elements[i];
	}

	return sparseMatrix;
}

float* generateMatrixSparseFromDense(float* denseMatrix, unsigned int* bitmap, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type) {
        std::vector<float> elements;
        if(gen_type==GEN_BY_ROWS) {
            for(int i=0; i<rows; i++) {
                for(int j=0; j<cols; j++) {
                    if(bitmap[i*cols+j]) {
                        elements.push_back(denseMatrix[i*cols+j]);
                    }
                }
            }
        }

        else { //In columns order (KN)
            for(int j=0; j<cols; j++) {
                for(int i=0; i<rows; i++) {
                    if(bitmap[i*cols+j]) {
                        elements.push_back(denseMatrix[i*cols+j]);
                    }
                }
            }
        }

        float* sparseMatrix = new float[elements.size()];
        for(int i=0; i<elements.size(); i++) {
            sparseMatrix[i]=elements[i];
        }

        return sparseMatrix;
}


/////
int* generateMinorIDFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, int &nnz, GENERATION_TYPE gen_type) {
	std::vector<int> elements;
        if(gen_type==GEN_BY_ROWS) {			//we need col_id
            for(int i=0; i<rows; i++) {
                int non_zeros=0;
                for(int j=0; j<cols; j++) {
                    if(denseMatrix[i*cols+j]!=0.0) {
                        elements.push_back(j);
                        non_zeros++;
		    }
                }
                if(non_zeros==0)
                	elements.push_back(0);
	    }
	}

        else { //In columns order (KN)			//we need row_id
            for(int j=0; j<cols; j++) {
            int non_zeros=0;
                for(int i=0; i<rows; i++) {
                    if(denseMatrix[i*cols+j]!=0.0) {
                        elements.push_back(i);
                        non_zeros++;
		    }
		}
		if(non_zeros==0)
			elements.push_back(0);
            }
	}	

	int* minor_id = new int[elements.size()];
	nnz=elements.size();
	for(int i=0; i<elements.size(); i++) {
            minor_id[i]=elements[i];
	}

	return minor_id;
}

/////
//int* generateMajorIDFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type) {
//	std::vector<int> elements;
//        if(gen_type==GEN_BY_ROWS) {			//we need row_id
//            for(int i=0; i<rows; i++) {
//                for(int j=0; j<cols; j++) {
//                    if(denseMatrix[i*cols+j]!=0.0) {
//                        elements.push_back(i);
//		    }
//                }
//	    }
//	}

//        else { //In columns order (KN)			//we need col_id
//            for(int j=0; j<cols; j++) {
//                for(int i=0; i<rows; i++) {
//                    if(denseMatrix[i*cols+j]!=0.0) {
//                        elements.push_back(j);
//		    }
//		}
//            }
//	}	

//	int* minor_id = new int[elements.size()];
//	for(int i=0; i<elements.size(); i++) {
//            minor_id[i]=elements[i];
//	}

//	return minor_id;
//}


/////
int* generateMajorPointerFromDense(float* denseMatrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type) {
	std::vector<int> elements;
        if(gen_type==GEN_BY_ROWS) {			//we need row_ptr
        int nnzp=0;
            for(int i=0; i<rows; i++) {
            int flag=0;
                for(int j=0; j<cols; j++) {
                    if(denseMatrix[i*cols+j]!=0.0) {
                        if(!flag)
                        {
                        	elements.push_back(nnzp);
                        	flag=1;
                        }
                        nnzp++;
		    }
                }
                if(!flag)
                {
                	//nnzp++;
                	elements.push_back(nnzp);
                	nnzp++;
                }
	    }
	   elements.push_back(nnzp); 
	}

        else { //In columns order (KN)			//we need col_ptr
        int nnzp=0;
            for(int j=0; j<cols; j++) {
            int flag=0;
                for(int i=0; i<rows; i++) {
                    if(denseMatrix[i*cols+j]!=0.0) {
                    	if(!flag)
                    	{
                        elements.push_back(nnzp);
                        flag=1;
                        }
                        nnzp++;
		    }
		}
		if(!flag)
		{	//nnzp++;
                	elements.push_back(nnzp);
                	nnzp++;
                }
            }
            elements.push_back(nnzp);
	}	

	int* major_pointer = new int[elements.size()];
	for(int i=0; i<elements.size(); i++) {
            major_pointer[i]=elements[i];
	}

	return major_pointer;
}


void printDenseMatrix(float* matrix, unsigned int rows, unsigned int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            std::cout << matrix[i*cols+j] << " ";
	}
	std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printBitMap(unsigned int* bitmap, unsigned int rows, unsigned int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            std::cout << bitmap[i*cols+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

}

void printSparseMatrix(float* sparseMatrix, unsigned int* bitmap, unsigned int rows, unsigned int cols) {
    unsigned int n_elements = 0;
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            if(bitmap[i*cols+j]) {
                n_elements++;
	    }
        }
    }

    for(int i=0; i<n_elements; i++) {
        std::cout << sparseMatrix[i] << " ";
    }
    std::cout << std::endl;

}


//Ordering-based optimization functions
//
//

void organizeMatrix (float* matrix, unsigned int rows, unsigned int cols, unsigned int* pointer_table, GENERATION_TYPE gen_type) {

    //Creating a copy
    float* matrix_copy = new float[rows*cols];
    for(int i=0; i<rows*cols; i++) {
        matrix_copy[i]=matrix[i];
    }

    if(gen_type == GEN_BY_ROWS) {
        for(int i=0; i<rows; i++) {
            int cluster_src = pointer_table[i];
            for(int j=0; j<cols; j++) {
                matrix[i*cols+j]=matrix_copy[cluster_src*cols+j];
            }
        }
    }

    else { //GEN BY COLS
       for(int j=0; j<cols; j++) {
            int cluster_src = pointer_table[j];
            for(int i=0; i<rows; i++) {
                matrix[i*cols+j]=matrix_copy[i*cols+cluster_src];
            }
        }
    }

     

    delete[] matrix_copy;
}


void organizeMatrixBack (float* matrix, unsigned int rows, unsigned int cols, unsigned int* pointer_table, GENERATION_TYPE gen_type) {
    float* matrix_copy = new float[rows*cols];
    for(int i=0; i<rows*cols; i++) {
        matrix_copy[i]=matrix[i];
    }

    if(gen_type == GEN_BY_ROWS) {
        for(int i=0; i<rows; i++) {
            int cluster_dst = pointer_table[i];
            for(int j=0; j<cols; j++) {
                matrix[cluster_dst*cols+j]=matrix_copy[i*cols+j];
            }
        }
    }

    else {  //GEN BY COLS
        for(int j=0; j<cols; j++) {
            int cluster_dst = pointer_table[j];
            for(int i=0; i<rows; i++) {
                matrix[i*cols+cluster_dst]=matrix_copy[i*cols+j];
            }
        }
    }

    delete[] matrix_copy;

}

/*
unsigned int* calculateOrdering (float* matrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type) {
    //calculating first the cluster size of each row
    unsigned int* pointer_table;
    unsigned int* size_rows;
    if(gen_type==GEN_BY_ROWS) {
        pointer_table = new unsigned int[rows];
        size_rows = new unsigned int [rows];
    
        for(int i=0; i<rows; i++) {
           pointer_table[i]=i;
           size_rows[i] = 0;
           for(int j=0; j<cols; j++) {
               if(matrix[i*cols+j] != 0.0) {
                   size_rows[i]++;
               }
           }
        }
   }

    else {
        pointer_table = new unsigned int[cols];
        size_rows = new unsigned int [cols];
    
        for(int j=0; j<cols; j++) {
           pointer_table[j]=j;
           size_rows[j] = 0;
           for(int i=0; i<rows; i++) {
               if(matrix[i*cols+j] != 0.0) {
                   size_rows[j]++;
               }
           }

        }

    }

    //Ordenamos y al mismo tiempo vamos modificando el indice de la matriz
    //We use bubble algorithm for being fast (the number of rows is not gonna be large)
    int dim_table;
    if(gen_type==GEN_BY_ROWS) {
        dim_table = rows;
    }

    else {
        dim_table = cols;
    }
    for(int i=0; i<dim_table; i++) {
        for(int j=i+1; j<dim_table; j++) {
            if(size_rows[j] > size_rows[i]) {

                int temp = size_rows[i];
                size_rows[i] = size_rows[j];
                size_rows[j] = temp;
                int row_temp = pointer_table[i];
                pointer_table[i]=pointer_table[j];
                pointer_table[j] = row_temp;
            }
        }
    }

    delete[] size_rows;
    return pointer_table;
}
*/
unsigned int* calculateOrdering (float* matrix, unsigned int rows, unsigned int cols, GENERATION_TYPE gen_type, int num_ms) {
    //calculating first the cluster size of each row
    unsigned int* pointer_table;
    unsigned int* size_rows;
    if(gen_type==GEN_BY_ROWS) {
        pointer_table = new unsigned int[rows];
        size_rows = new unsigned int [rows];

        for(int i=0; i<rows; i++) {
           pointer_table[i]=i;
           size_rows[i] = 0;
           for(int j=0; j<cols; j++) {
               if(matrix[i*cols+j] != 0.0) {
                   size_rows[i]++;
               }
           }
	   if(size_rows[i]<3) {
               size_rows[i]=3; //Minimum cluster size
	   }
        }
   }

    else {
        pointer_table = new unsigned int[cols];
        size_rows = new unsigned int [cols];

        for(int j=0; j<cols; j++) {
           pointer_table[j]=j;
           size_rows[j] = 0;
           for(int i=0; i<rows; i++) {
               if(matrix[i*cols+j] != 0.0) {
                   size_rows[j]++;
               }
           }

	   if(size_rows[j] < 3) {
               size_rows[j] = 3;
	   }

        }

    }

    //Ordenamos y al mismo tiempo vamos modificando el indice de la matriz
    //We use bubble algorithm for being fast (the number of rows is not gonna be large)
    int dim_table;
    if(gen_type==GEN_BY_ROWS) {
        dim_table = rows;
    }

    else {
        dim_table = cols;
    }
    unsigned int n_rows_selected = 0;
  
    //First the ones that does not fit 
    for(int i=0; i<dim_table; i++) {
        if(size_rows[i] > num_ms) {
            int temp = size_rows[i];
	    size_rows[i] = size_rows[n_rows_selected];
	    size_rows[n_rows_selected] = temp;
            int row_temp = pointer_table[i];
	    pointer_table[i] = pointer_table[n_rows_selected];
	    pointer_table[n_rows_selected] = row_temp;
	    n_rows_selected++;
	}
    }
    unsigned int n_ms_used = 0;
    bool row_found;
    unsigned int greater_row; 
    while(n_rows_selected < dim_table) {
        row_found = false;
	for(int i=n_rows_selected; i < dim_table; i++) {
	    if((size_rows[i] + n_ms_used) <= num_ms) {
		if(row_found == false) {
                    row_found = true;
                    greater_row = i;
		}

		else { //If it is true 
                    if(size_rows[i] > size_rows[greater_row]) {
                        greater_row = i;
		    }

		}
	    }
	}

	if(row_found) {
            int temp = size_rows[greater_row];
	    n_ms_used+=temp;
            size_rows[greater_row] = size_rows[n_rows_selected];
            size_rows[n_rows_selected] = temp;
            int row_temp = pointer_table[greater_row];
            pointer_table[greater_row] = pointer_table[n_rows_selected];
            pointer_table[n_rows_selected] = row_temp;
            n_rows_selected++;


	}

	else {
            n_ms_used = 0; 
	}
    }


    delete[] size_rows;
    return pointer_table;
}

