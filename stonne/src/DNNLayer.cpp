// Created by Francisco Munoz Martinez on 02/07/2019

#include "DNNLayer.h"
#include "utility.h"

DNNLayer::DNNLayer(Layer_t layer_type, std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G, unsigned int N, unsigned int X, unsigned int Y, unsigned int strides) {
    this->R = R;
    this->S = S;
    this->C = C / G; //The user has to specify this parameter in terms of the whole feature map
    this->K = K / G;  //Idem 
    this->G = G;
    this->N = N;
    this->X = X;
    this->Y = Y;
    this->strides = strides;
    this->layer_name = layer_name;
    this->layer_type = layer_type;

    this->X_ = (X - R + strides) / strides;
    this->Y_ = (Y - S + strides) / strides;
   
}

void DNNLayer::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"LayerConfiguration\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"Layer_Type\" : " << this->layer_type << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"R\" : " << this->R << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"S\" : " << this->S << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"C\" : " << this->C << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"K\" : " << this->K << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"G\" : " << this->G << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"N\" : " << this->N << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"X\" : " << this->X << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"Y\" : " << this->Y << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"X_\" : " << this->X_ << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"Y_\" : " << this->Y_  << std::endl;
    out << ind(indent) << "}";
}
