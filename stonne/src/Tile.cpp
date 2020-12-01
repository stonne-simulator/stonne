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

    


Tile::Tile(std::string tile_file) {
    auto config = cpptoml::parse_file(tile_file); //Creating object to parse
    auto tile_type=config->get_as<std::string>("tile_type");
    auto T_R=config->get_as<int32_t>("T_R");
    auto T_S=config->get_as<int32_t>("T_S");
    auto T_C=config->get_as<int32_t>("T_C");
    auto T_K=config->get_as<int32_t>("T_K");
    auto T_G=config->get_as<int32_t>("T_G");
    auto T_N=config->get_as<int32_t>("T_N");
    auto T_X_=config->get_as<int32_t>("T_X'");
    auto T_Y_=config->get_as<int32_t>("T_Y'");

    if(!tile_type) {
        std::cout << "Error to parse tile_type. Parameter not found" << std::endl;
        exit(1);
    }

    if(*tile_type=="CONV") { //Actually the architecture does not know about the layer type. This is just to make sure that the user introduces the 
    //appropiate parameters. 
        std::cout << "Reading a tile of type CONV" << std::endl; 
    }

    else if(*tile_type=="FC") {
        std::cout << "Reading a tile of type FC" << std::endl;
    }

    else {
        std::cout << "Error to parse tile_type. Specify a correct type: [CONV, FC, POOL]" << std::endl;
        exit(1);
    }

    if(*tile_type=="CONV") {
        if(!T_R) {
            std::cout << "Error to parse T_R. Value not found." << std::endl;
            exit(1);
        }

        if(!T_S) {
            std::cout << "Error to parse T_S. Value not found." << std::endl;
            exit(1);
        }

        if(!T_C) {
            std::cout << "Error to parse T_C. Value not found." << std::endl;
            exit(1);
        }

        if(!T_K) {
            std::cout << "Error to parse T_K. Value not found." << std::endl;
            exit(1);
        }

        if(!T_G) {
            std::cout << "Error to parse T_G. Value not found." << std::endl;
            exit(1);
        }

        if(!T_N) {
            std::cout << "Error to parse T_N. Value not found." << std::endl;
            exit(1);
        }

        if(!T_X_) {
            std::cout << "Error to parse T_X'. Value not found." << std::endl;
            exit(1);
        }

        if(!T_Y_) {
            std::cout << "Error to parse T_Y'. Value not found." << std::endl;
            exit(1);
        }



    //Filling the parameters
        this->T_R = *T_R;
        this->T_S = *T_S;
        this->T_C = *T_C;
        this->T_K = *T_K;
        this->T_G = *T_G;
        this->T_N = *T_N;
        this->T_X_ = *T_X_;
        this->T_Y_ = *T_Y_;

    }

    else if(*tile_type=="FC") {

         if(!T_N) {
            std::cout << "Error to parse T_N. Value not found." << std::endl;
            exit(1);
          }

         if(!T_S) {
            std::cout << "Error to parse T_S. Value not found." << std::endl;
            exit(1);
         }

         if(!T_K) {
            std::cout << "Error to parse T_K. Value not found." << std::endl;
            exit(1);
         }

         //Filling the parameters
         this->T_R = 1;
         this->T_S=*T_S;
         this->T_C=1;
         this->T_K=*T_K;
         this->T_G=1;
         this->T_N=*T_N;
         this->T_X_=1;
         this->T_Y_=1;
     
        

    }
   
    //Folding is not specified in this case since this use case is not to load the tile into the architecture. Rather, it is to load the tile from the file and layer specify all the parameters
    // to the architecture by means of some abstractions like an instruction.


 
} //End constructor



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





