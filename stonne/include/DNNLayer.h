// Created by Francisco Munoz Martinez on 02/07/2019

#ifndef __DNN_LAYER__H
#define __DNN_LAYER__H

#include "types.h"
#include <iostream>

class DNNLayer {
private:
    Layer_t layer_type;
    std::string layer_name;        // Layer name used to create the output file
    unsigned int R;           // Number of Filter Rows
    unsigned int S;           // Number of filter columns
    unsigned int C;           // Number of filter and input channels
    unsigned int K;           // Number of filters and output channels per group
    unsigned int G;           // Number of grups 
    unsigned int N;           // Number of inputs (batch size)
    unsigned int X;           // Number of input fmap rows
    unsigned int Y;           // Number of input fmap columns
    unsigned int X_;          // Number of output fmap rows
    unsigned int Y_;          // Number of output fmap columns
    unsigned int strides;     // Strides
  



public: 
    //K = Number of total filters in the network. C= Number  of input channels (the whole feature map). G=Number of groups
    DNNLayer(Layer_t layer_type, std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides);
    unsigned int get_R()          const {return this->R;}
    unsigned int get_S()          const {return this->S;}
    unsigned int get_C()          const {return this->C;}
    unsigned int get_K()          const {return this->K;}
    unsigned int get_G()          const {return this->G;}
    unsigned int get_N()          const {return this->N;}
    unsigned int get_X()          const {return this->X;}
    unsigned int get_Y()          const {return this->Y;}
    unsigned int get_X_()         const {return this->X_;}
    unsigned int get_Y_()         const {return this->Y_;}
    unsigned int get_strides()    const {return this->strides;}
    std::string get_name()        const {return this->layer_name;}
    Layer_t get_layer_type()      const {return this->layer_type;}

    void printConfiguration(std::ofstream& out, unsigned int indent);
};

#endif

