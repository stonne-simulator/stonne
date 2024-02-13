#ifndef __DNN_LAYER__H
#define __DNN_LAYER__H

#include <iostream>
#include "../types.hpp"

class DNNLayer {
 private:
  Layer_t layer_type;
  std::string layer_name;  // Layer name used to create the output file
  std::size_t R;           // Number of Filter Rows
  std::size_t S;           // Number of filter columns
  std::size_t C;           // Number of filter and input channels
  std::size_t K;           // Number of filters and output channels per group
  std::size_t G;           // Number of grups
  std::size_t N;           // Number of inputs (batch size)
  std::size_t X;           // Number of input fmap rows
  std::size_t Y;           // Number of input fmap columns
  std::size_t X_;          // Number of output fmap rows
  std::size_t Y_;          // Number of output fmap columns
  std::size_t strides;     // Strides

 public:
  //K = Number of total filters in the network. C= Number  of input channels (the whole feature map). G=Number of groups
  DNNLayer(Layer_t layer_type, std::string layer_name, std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X,
           std::size_t Y, std::size_t strides);

  std::size_t get_R() const { return this->R; }

  std::size_t get_S() const { return this->S; }

  std::size_t get_C() const { return this->C; }

  std::size_t get_K() const { return this->K; }

  std::size_t get_G() const { return this->G; }

  std::size_t get_N() const { return this->N; }

  std::size_t get_X() const { return this->X; }

  std::size_t get_Y() const { return this->Y; }

  std::size_t get_X_() const { return this->X_; }

  std::size_t get_Y_() const { return this->Y_; }

  std::size_t get_strides() const { return this->strides; }

  std::string get_name() const { return this->layer_name; }

  Layer_t get_layer_type() const { return this->layer_type; }

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

#endif
