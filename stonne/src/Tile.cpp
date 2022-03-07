//Created by Francisco Munoz Martinez on 26/06/2019

#include "Tile.h"
#include "utility.h"
#include <math.h>
#include "types.h"
#include <assert.h>
#include "cpptoml.h"




//Used to create a convolutional tile
Tile::Tile(unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G, unsigned int T_N, unsigned int T_X_, unsigned int T_Y_, bool folding) {
    this->T_R = T_R;
    this->T_S = T_S; 
    this->T_C = T_C;
    this->T_K = T_K;
    this->T_G = T_G;
    this->T_N = T_N;
    this->T_X_ = T_X_;
    this->T_Y_ = T_Y_;
          
    this->VN_Size = T_R*T_S*T_C;
    this->Num_VNs = T_K*T_G*T_N*T_X_*T_Y_;
    this->folding = folding;
    if(this->folding) {
        this->VN_Size+=1; //1 MS extra to psum accumulation
    }
}

// Used to create a fully connected tile
Tile::Tile(unsigned int T_M, unsigned int T_N, unsigned int T_K, bool folding) {
    this->T_R = 1;
    this->T_S = T_K;
    this->T_C = 1;
    this->T_K = T_N;
    this->T_G = 1;
    this->T_N = 1;
    this->T_X_ = T_M;
    this->T_Y_ = 1;

    this->VN_Size = T_R*T_S*T_C;
    this->Num_VNs = T_K*T_G*T_N*T_X_*T_Y_;
    this->folding = folding;
    if(this->folding) {
        this->VN_Size+=1; //1 MS extra to psum accumulation
    }
}


void Tile::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"TileConfiguration\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_R\" : " << this->T_R << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_S\" : " << this->T_S << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_C\" : " << this->T_C << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_K\" : " << this->T_K << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_G\" : " << this->T_G << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_N\" : " << this->T_N << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_X_\" : " << this->T_X_ << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"T_Y_\" : " << this->T_Y_ << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"VN_Size\" : " << this->VN_Size << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"Num_VNs\" : " << this->Num_VNs << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"folding_enabled\" : " << this->folding  << std::endl;
        //out << ind(indent+IND_SIZE) << "\"n_folding\" : " << this->n_folding  << std::endl;
        
    out << ind(indent) << "}";
}





