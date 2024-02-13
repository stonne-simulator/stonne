#include "utility.hpp"
#include <assert.h>
#include <math.h>
#include <algorithm>
#include <vector>

bool isNum(std::string str) {
  std::istringstream sin(str);
  double d;
  char c;
  if (!(sin >> d)) {
    return false;
  } else if (sin >> c) {
    return false;
  } else {
    return true;
  }
}

std::string getstr(std::istringstream& instr) {
  std::string str;
  while (instr >> str) {
    if (isNum(str)) {
      return str;
    } else {
      continue;
    }
  }
  return str;
}

bool ispowerof2(std::size_t x) {
  return x && !(x & (x - 1));
}

std::string get_string_layer_t(Layer_t kernelOperation) {
  switch (kernelOperation) {
    case CONV:
      return "CONV";
      break;
    case GEMM:
      return "GEMM";
      break;
    case bitmapSpMSpM:
      return "bitmapSpMSpM";

    case csrSpMM:
      return "csrSpMM";
    case innerProductGEMM:
      return "innerProductGEMM";
    case outerProductGEMM:
      return "outerProductGEMM";
    case gustavsonsGEMM:
      return "gustavsonsGEMM";
    default:
      assert(false);
      break;
  }
}

std::size_t nextPowerOf2(int x) {
  return pow(2, ceil(log2(x)));
}

std::string to_lower(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  return str;
}

TileGenerator::Target parseTileGeneratorTarget(std::string target) {
  target = to_lower(target);
  if (target == "none" || target == "0")
    return TileGenerator::Target::NONE;
  else if (target == "performance" || target == "1")
    return TileGenerator::Target::PERFORMANCE;
  else if (target == "energy" || target == "2")
    return TileGenerator::Target::ENERGY;
  else if (target == "energy_efficiency" || target == "3")
    return TileGenerator::Target::ENERGY_EFFICIENCY;

  std::cerr << "TileGenerator target <" << target << "> is not recognized" << std::endl;
  std::cerr << "-generate_tile only supports 0 (none), 1 (performance), 2 (energy) or 3 (energy_efficiency)" << std::endl;
  std::cerr << "Changing TileGenerator target to Target::None" << std::endl;

  return TileGenerator::Target::NONE;
}

std::string parseTileGeneratorTarget(TileGenerator::Target target) {
  switch (target) {
    case TileGenerator::Target::NONE:
      return "none";
    case TileGenerator::Target::PERFORMANCE:
      return "performance";
    case TileGenerator::Target::ENERGY:
      return "energy";
    case TileGenerator::Target::ENERGY_EFFICIENCY:
      return "energy_efficiency";
  }
  return "";
}

Layer_t get_type_layer_t(std::string kernelOperation) {
  if (kernelOperation == "CONV") {
    return CONV;
  } else if (kernelOperation == "GEMM") {
    return GEMM;
  }

  else if (kernelOperation == "bitmapSpMSpM") {
    return bitmapSpMSpM;
  }

  else if (kernelOperation == "csrSpMM") {
    return csrSpMM;
  }

  else if (kernelOperation == "innerProductGEMM") {
    return innerProductGEMM;
  }

  else if (kernelOperation == "outerProductGEMM") {
    return outerProductGEMM;
  }

  else if (kernelOperation == "gustavsonsGEMM") {
    return gustavsonsGEMM;
  }

  else {
    std::cout << kernelOperation << " Not found" << std::endl;
    assert(false);
  }
}

TileGenerator::Generator parseTileGenerator(std::string generator) {
  generator = to_lower(generator);
  if (generator == "auto" || generator == "0")
    return TileGenerator::Generator::CHOOSE_AUTOMATICALLY;
  else if (generator == "mrna" || generator == "1")
    return TileGenerator::Generator::MRNA;
  else if (generator == "stonnemapper" || generator == "2")
    return TileGenerator::Generator::STONNE_MAPPER;

  std::cerr << "TileGenerator generator <" << generator << "> is not recognized" << std::endl;
  std::cerr << "-generator only supports 0 (automatic), 1 (mRNA) or 2 (StonneMapper)" << std::endl;
  std::cerr << "Changing TileGenerator generator to Generator::CHOOSE_AUTOMATICALLY" << std::endl;

  return TileGenerator::Generator::CHOOSE_AUTOMATICALLY;
}

std::string parseTileGenerator(TileGenerator::Generator generator) {
  switch (generator) {
    case TileGenerator::Generator::CHOOSE_AUTOMATICALLY:
      return "Auto";
    case TileGenerator::Generator::MRNA:
      return "mRNA";
    case TileGenerator::Generator::STONNE_MAPPER:
      return "StonneMapper";
  }
  return "";
}

std::string get_string_adder_configuration(adderconfig_t config) {
  switch (config) {
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

    case SORT_TREE:
      return "SORT_TREE";

    default:
      assert(false);
  }
}

std::string get_string_fwlink_direction(fl_t fl_direction) {
  switch (fl_direction) {
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
  switch (reduce_network_type) {
    case ASNETWORK:
      return "ASNETWORK";
      break;
    case FENETWORK:
      return "FENETWORK";
      break;
    case TEMPORALRN:
      return "TEMPORALRN";
      break;
    case SPARSEFLEX_MERGER:
      return "SPARSEFLEX_MERGER";
      break;
    default:
      assert(false);
      break;
  }
}

ReduceNetwork_t get_type_reduce_network_type(std::string reduce_network_type) {
  if (reduce_network_type == "ASNETWORK") {
    return ASNETWORK;
  } else if (reduce_network_type == "FENETWORK") {
    return FENETWORK;
  }

  else if (reduce_network_type == "TEMPORALRN") {
    return TEMPORALRN;
  }

  else if (reduce_network_type == "SPARSEFLEX_MERGER") {
    return SPARSEFLEX_MERGER;
  } else {
    std::cout << reduce_network_type << " Not found" << std::endl;
    assert(false);
  }
}

std::string get_string_multiplier_network_type(MultiplierNetwork_t multiplier_network_type) {
  switch (multiplier_network_type) {
    case LINEAR:
      return "LINEAR";
      break;
    case OS_MESH:
      return "OS_MESH";
      break;
    case SPARSEFLEX_LINEAR:
      return "SPARSEFLEX_LINEAR";
    default:
      assert(false);
      break;
  }
}

MultiplierNetwork_t get_type_multiplier_network_type(std::string multiplier_network_type) {
  if (multiplier_network_type == "LINEAR") {
    return LINEAR;
  } else if (multiplier_network_type == "OS_MESH") {
    return OS_MESH;
  }

  else if (multiplier_network_type == "SPARSEFLEX_LINEAR") {
    return SPARSEFLEX_LINEAR;
  }

  else {
    std::cout << multiplier_network_type << " Not found" << std::endl;
    assert(false);
  }
}

std::string get_string_memory_controller_type(MemoryController_t memory_controller_type) {
  switch (memory_controller_type) {
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

    case MAGMA_SPARSE_DENSE:
      return "MAGMA_SPARSE_DENSE";
      break;

    case OUTER_PRODUCT_GEMM:
      return "OUTER_PRODUCT_GEMM";
    case GUSTAVSONS_GEMM:
      return "GUSTAVSONS_GEMM";

    default:
      assert(false);
      break;
  }
}

MemoryController_t get_type_memory_controller_type(std::string memory_controller_type) {
  if (memory_controller_type == "MAERI_DENSE_WORKLOAD") {
    return MAERI_DENSE_WORKLOAD;
  } else if (memory_controller_type == "SIGMA_SPARSE_GEMM") {
    return SIGMA_SPARSE_GEMM;
  }
  /////

  else if (memory_controller_type == "TPU_OS_DENSE") {
    return TPU_OS_DENSE;
  }

  else if (memory_controller_type == "MAGMA_SPARSE_DENSE") {
    return MAGMA_SPARSE_DENSE;
  }

  else if (memory_controller_type == "OUTER_PRODUCT_GEMM") {
    return OUTER_PRODUCT_GEMM;
  }

  else if (memory_controller_type == "GUSTAVSONS_GEMM") {
    return GUSTAVSONS_GEMM;
  }

  else {
    std::cout << memory_controller_type << " Not found" << std::endl;
    assert(false);
  }
}

Dataflow get_type_dataflow_type(std::string dataflow_type) {
  if (dataflow_type == "CNN_DATAFLOW") {
    return CNN_DATAFLOW;
  } else if (dataflow_type == "MK_STA_KN_STR") {
    return MK_STA_KN_STR;
  }

  else if (dataflow_type == "MK_STR_KN_STA") {
    return MK_STR_KN_STA;
  }
  /////
  else if (dataflow_type == "SPARSE_DENSE_DATAFLOW") {
    return SPARSE_DENSE_DATAFLOW;
  }

  else {
    std::cout << dataflow_type << " Not found" << std::endl;
    assert(false);
  }
}

std::string get_string_dataflow_type(Dataflow dataflow) {
  switch (dataflow) {
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

std::string ind(std::size_t indent) {
  std::string str = "";
  for (int i = 0; i < indent; i++) {
    str += " ";
  }

  return str;
}

float* generatePrunnedMatrix(const float* src_matrix, std::size_t size, float pr_ratio) {
  float* dst_matrix = new float[size];
  for (int i = 0; i < size; i++) {
    dst_matrix[i] = fabs(src_matrix[i]);
  }
  int n = size * pr_ratio;
  std::cout << "n to crib is " << n << std::endl;
  float* begin = dst_matrix;
  float* end = begin + size;
  float* nth = begin + n;
  std::nth_element(begin, nth, end);  //Sorting
  float pivot = dst_matrix[n];
  //Copying and prunning
  for (int i = 0; i < size; i++) {
    float value_abs = fabs(src_matrix[i]);
    if (value_abs < pivot) {
      dst_matrix[i] = 0.0;  //Prunned
    } else {
      dst_matrix[i] = src_matrix[i];  //Not prunned
    }
  }

  return dst_matrix;
}

float* generateMatrixDense(std::size_t rows, std::size_t cols, std::size_t sparsity) {
  float* matrix = new float[rows * cols];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float value;
      std::size_t sparse_yes = rand() % 100;
      if (sparse_yes <= sparsity) {
        value = 0.0;
      }

      else {
        value = rand() % 10 + 1;
      }

      matrix[i * cols + j] = value;
    }
  }

  return matrix;
}

std::size_t* generateBitMapFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type) {
  std::size_t* bitMap = new std::size_t[rows * cols];
  if (gen_type == GEN_BY_ROWS) {
    std::size_t non_zeros = 0;
    for (int i = 0; i < rows; i++) {
      non_zeros = 0;
      for (int j = 0; j < cols; j++) {
        if (denseMatrix[i * cols + j] != 0.0) {
          bitMap[i * cols + j] = 1;
          non_zeros++;
        }

        else {
          bitMap[i * cols + j] = 0;
        }
      }

      if (non_zeros < 3) {
        for (int j = 0; (j < 3) && (non_zeros < 3); j++) {
          if (bitMap[i * cols + j] == 0) {
            bitMap[i * cols + j] = 1;  //Even if this is a 0. This is to make sure sizes greater than 3
            non_zeros++;
          }
        }
      }
    }
  }

  else if (gen_type == GEN_BY_COLS) {
    std::size_t non_zeros = 0;
    for (int j = 0; j < cols; j++) {
      non_zeros = 0;
      for (int i = 0; i < rows; i++) {
        if (denseMatrix[i * cols + j] != 0.0) {  //TODO Warning with the accuracy
          bitMap[i * cols + j] = 1;
          non_zeros++;
        }

        else {
          bitMap[i * cols + j] = 0;
        }
      }

      if (non_zeros < 3) {
        for (int i = 0; (i < 3) && (non_zeros < 3); i++) {
          if (bitMap[i * cols + j] == 0) {
            bitMap[i * cols + j] = 1;  //Even if this is a 0. This is to make sure sizes greater than 3
            non_zeros++;
          }
        }
      }
    }
  }

  return bitMap;
}

/////
float* generateMatrixSparseFromDenseNoBitmap(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type, std::size_t& size) {
  std::vector<float> elements;
  if (gen_type == GEN_BY_ROWS) {
    for (int i = 0; i < rows; i++) {
      int non_zeros = 0;
      for (int j = 0; j < cols; j++) {
        if (denseMatrix[i * cols + j] != 0.0) {
          elements.push_back(denseMatrix[i * cols + j]);
          non_zeros++;
        }
      }
      if (non_zeros == 0)
        elements.push_back(0.0);
    }
  }

  else {  //In columns order (KN)
    for (int j = 0; j < cols; j++) {
      int non_zeros = 0;
      for (int i = 0; i < rows; i++) {
        if (denseMatrix[i * cols + j] != 0.0) {
          elements.push_back(denseMatrix[i * cols + j]);
          non_zeros++;
        }
      }
      if (non_zeros == 0)
        elements.push_back(0.0);
    }
  }

  float* sparseMatrix = new float[elements.size()];
  for (int i = 0; i < elements.size(); i++) {
    sparseMatrix[i] = elements[i];
  }

  size = elements.size();
  return sparseMatrix;
}

float* generateMatrixSparseFromDense(float* denseMatrix, std::size_t* bitmap, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type, std::size_t& size) {
  std::vector<float> elements;
  if (gen_type == GEN_BY_ROWS) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (bitmap[i * cols + j]) {
          elements.push_back(denseMatrix[i * cols + j]);
        }
      }
    }
  }

  else {  //In columns order (KN)
    for (int j = 0; j < cols; j++) {
      for (int i = 0; i < rows; i++) {
        if (bitmap[i * cols + j]) {
          elements.push_back(denseMatrix[i * cols + j]);
        }
      }
    }
  }

  float* sparseMatrix = new float[elements.size()];
  for (int i = 0; i < elements.size(); i++) {
    sparseMatrix[i] = elements[i];
  }

  size = elements.size();
  return sparseMatrix;
}

/////
int* generateMinorIDFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, int& nnz, GENERATION_TYPE gen_type) {
  std::vector<int> elements;
  if (gen_type == GEN_BY_ROWS) {  //we need col_id
    for (int i = 0; i < rows; i++) {
      int non_zeros = 0;
      for (int j = 0; j < cols; j++) {
        if (denseMatrix[i * cols + j] != 0.0) {
          elements.push_back(j);
          non_zeros++;
        }
      }
      if (non_zeros == 0)
        elements.push_back(0);
    }
  }

  else {  //In columns order (KN)			//we need row_id
    for (int j = 0; j < cols; j++) {
      int non_zeros = 0;
      for (int i = 0; i < rows; i++) {
        if (denseMatrix[i * cols + j] != 0.0) {
          elements.push_back(i);
          non_zeros++;
        }
      }
      if (non_zeros == 0)
        elements.push_back(0);
    }
  }

  int* minor_id = new int[elements.size()];
  nnz = elements.size();
  for (int i = 0; i < elements.size(); i++) {
    minor_id[i] = elements[i];
  }

  return minor_id;
}

/////
//int* generateMajorIDFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type) {
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
int* generateMajorPointerFromDense(float* denseMatrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type) {
  std::vector<int> elements;
  if (gen_type == GEN_BY_ROWS) {  //we need row_ptr
    int nnzp = 0;
    for (int i = 0; i < rows; i++) {
      int flag = 0;
      for (int j = 0; j < cols; j++) {
        if (denseMatrix[i * cols + j] != 0.0) {
          if (!flag) {
            elements.push_back(nnzp);
            flag = 1;
          }
          nnzp++;
        }
      }
      if (!flag) {
        //nnzp++;
        elements.push_back(nnzp);
        nnzp++;
      }
    }
    elements.push_back(nnzp);
  }

  else {  //In columns order (KN)			//we need col_ptr
    int nnzp = 0;
    for (int j = 0; j < cols; j++) {
      int flag = 0;
      for (int i = 0; i < rows; i++) {
        if (denseMatrix[i * cols + j] != 0.0) {
          if (!flag) {
            elements.push_back(nnzp);
            flag = 1;
          }
          nnzp++;
        }
      }
      if (!flag) {  //nnzp++;
        elements.push_back(nnzp);
        nnzp++;
      }
    }
    elements.push_back(nnzp);
  }

  int* major_pointer = new int[elements.size()];
  for (int i = 0; i < elements.size(); i++) {
    major_pointer[i] = elements[i];
  }

  return major_pointer;
}

void printDenseMatrix(float* matrix, std::size_t rows, std::size_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void printBitMap(std::size_t* bitmap, std::size_t rows, std::size_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << bitmap[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void printSparseMatrix(float* sparseMatrix, std::size_t* bitmap, std::size_t rows, std::size_t cols) {
  std::size_t n_elements = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (bitmap[i * cols + j]) {
        n_elements++;
      }
    }
  }

  for (int i = 0; i < n_elements; i++) {
    std::cout << sparseMatrix[i] << " ";
  }
  std::cout << std::endl;
}

//Ordering-based optimization functions
//
//

void organizeMatrix(float* matrix, std::size_t rows, std::size_t cols, std::size_t* pointer_table, GENERATION_TYPE gen_type) {

  //Creating a copy
  float* matrix_copy = new float[rows * cols];
  for (int i = 0; i < rows * cols; i++) {
    matrix_copy[i] = matrix[i];
  }

  if (gen_type == GEN_BY_ROWS) {
    for (int i = 0; i < rows; i++) {
      int cluster_src = pointer_table[i];
      for (int j = 0; j < cols; j++) {
        matrix[i * cols + j] = matrix_copy[cluster_src * cols + j];
      }
    }
  }

  else {  //GEN BY COLS
    for (int j = 0; j < cols; j++) {
      int cluster_src = pointer_table[j];
      for (int i = 0; i < rows; i++) {
        matrix[i * cols + j] = matrix_copy[i * cols + cluster_src];
      }
    }
  }

  delete[] matrix_copy;
}

void organizeMatrixBack(float* matrix, std::size_t rows, std::size_t cols, std::size_t* pointer_table, GENERATION_TYPE gen_type) {
  float* matrix_copy = new float[rows * cols];
  for (int i = 0; i < rows * cols; i++) {
    matrix_copy[i] = matrix[i];
  }

  if (gen_type == GEN_BY_ROWS) {
    for (int i = 0; i < rows; i++) {
      int cluster_dst = pointer_table[i];
      for (int j = 0; j < cols; j++) {
        matrix[cluster_dst * cols + j] = matrix_copy[i * cols + j];
      }
    }
  }

  else {  //GEN BY COLS
    for (int j = 0; j < cols; j++) {
      int cluster_dst = pointer_table[j];
      for (int i = 0; i < rows; i++) {
        matrix[i * cols + cluster_dst] = matrix_copy[i * cols + j];
      }
    }
  }

  delete[] matrix_copy;
}

/*
std::size_t* calculateOrdering (float* matrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type) {
    //calculating first the cluster size of each row
    std::size_t* pointer_table;
    std::size_t* size_rows;
    if(gen_type==GEN_BY_ROWS) {
        pointer_table = new std::size_t[rows];
        size_rows = new std::size_t [rows];
    
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
        pointer_table = new std::size_t[cols];
        size_rows = new std::size_t [cols];
    
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
std::size_t* calculateOrdering(float* matrix, std::size_t rows, std::size_t cols, GENERATION_TYPE gen_type, int num_ms) {
  //calculating first the cluster size of each row
  std::size_t* pointer_table;
  std::size_t* size_rows;
  if (gen_type == GEN_BY_ROWS) {
    pointer_table = new std::size_t[rows];
    size_rows = new std::size_t[rows];

    for (int i = 0; i < rows; i++) {
      pointer_table[i] = i;
      size_rows[i] = 0;
      for (int j = 0; j < cols; j++) {
        if (matrix[i * cols + j] != 0.0) {
          size_rows[i]++;
        }
      }
      if (size_rows[i] < 3) {
        size_rows[i] = 3;  //Minimum cluster size
      }
    }
  }

  else {
    pointer_table = new std::size_t[cols];
    size_rows = new std::size_t[cols];

    for (int j = 0; j < cols; j++) {
      pointer_table[j] = j;
      size_rows[j] = 0;
      for (int i = 0; i < rows; i++) {
        if (matrix[i * cols + j] != 0.0) {
          size_rows[j]++;
        }
      }

      if (size_rows[j] < 3) {
        size_rows[j] = 3;
      }
    }
  }

  //Ordenamos y al mismo tiempo vamos modificando el indice de la matriz
  //We use bubble algorithm for being fast (the number of rows is not gonna be large)
  int dim_table;
  if (gen_type == GEN_BY_ROWS) {
    dim_table = rows;
  }

  else {
    dim_table = cols;
  }
  std::size_t n_rows_selected = 0;

  //First the ones that does not fit
  for (int i = 0; i < dim_table; i++) {
    if (size_rows[i] > num_ms) {
      int temp = size_rows[i];
      size_rows[i] = size_rows[n_rows_selected];
      size_rows[n_rows_selected] = temp;
      int row_temp = pointer_table[i];
      pointer_table[i] = pointer_table[n_rows_selected];
      pointer_table[n_rows_selected] = row_temp;
      n_rows_selected++;
    }
  }
  std::size_t n_ms_used = 0;
  bool row_found;
  std::size_t greater_row;
  while (n_rows_selected < dim_table) {
    row_found = false;
    for (int i = n_rows_selected; i < dim_table; i++) {
      if ((size_rows[i] + n_ms_used) <= num_ms) {
        if (row_found == false) {
          row_found = true;
          greater_row = i;
        }

        else {  //If it is true
          if (size_rows[i] > size_rows[greater_row]) {
            greater_row = i;
          }
        }
      }
    }

    if (row_found) {
      int temp = size_rows[greater_row];
      n_ms_used += temp;
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

void sequential_layer(std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X, std::size_t Y,
                      std::size_t strides, float* input, float* filters, float* outputs) {

  std::size_t OX = (X - R + strides) / strides;
  std::size_t OY = (Y - S + strides) / strides;
  K /= G;
  C /= G;
  std::size_t output_size_n = G * K * OX * OY;
  std::size_t input_size_n = G * C * X * Y;
  std::size_t filter_size = R * S * C;
  std::size_t size_oy = OY * K * G;
  std::size_t size_y = Y * G * C;

  for (int n = 0; n < N; n++) {
    for (int g = 0; g < G; g++) {
      for (int k = 0; k < K; k++) {
        for (int ox = 0; ox < OX; ox++) {
          for (int oy = 0; oy < OY; oy++) {
            float& output = outputs[n * output_size_n + ox * size_oy + oy * K * G + g * K + k];
            output = 0.0;
            for (int c = 0; c < C; c++) {
              for (int r = 0; r < R; r++) {
                for (int s = 0; s < S; s++) {
                  output += input[n * input_size_n + ox * strides * size_y + oy * strides * C * G + r * size_y + s * C * G + g * C + c] *
                            filters[g * K * filter_size + k * filter_size + r * S * C + s * C + c];
                }
              }
            }
          }
        }
      }
    }
  }
}

void cpu_gemm(float* MK_dense_matrix, float* KN_dense_matrix, float* output, std::size_t M, std::size_t N, std::size_t K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float suma = 0;
      for (int k = 0; k < K; k++) {
        suma += MK_dense_matrix[i * K + k] * KN_dense_matrix[k * N + j];
      }

      output[i * N + j] = suma;
    }
  }
}
