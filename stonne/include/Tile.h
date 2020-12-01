//Created by Francisco Munoz Martinez on 26/06/2019

#ifndef __TILE__H
#define __TILE__H

#include <map>
#include "types.h"
#include <iostream>


/*
 * This class represents a sparse cluster mapped onto the architecture. 
 * Basically, size is the number of multipliers needed, and folding indicates if this cluster
 * requires an  extra multiplier to act as a forwarder.
 */
class SparseVN {
private:
    unsigned int size;
    bool folding;

public:
    SparseVN(unsigned int size, bool folding) {this->size=size; this->folding=folding;}
    unsigned int  get_VN_Size() {if(this->folding) {return this->size+1;} else {return this->size;}}
    bool getFolding() {return this->folding;}
};




/*
 This class represent a tile 
*/


class Tile {
private:
    unsigned int T_R;         // Number of filter rows
    unsigned int T_S;         // Number of filter columns
    unsigned int T_C;         // Number of input and filter channels
    unsigned int T_K;         // Number of filters and number of ofmap channels per group
    unsigned int T_G;         // Number of groups 
    unsigned int T_N;         // Batch size 
    unsigned int T_X_;        // Number of output fmap rows
    unsigned int T_Y_;        // Number of output fmap columns
    unsigned int VN_Size;     // Virtual Neuron Size (i.e., T_R*T_S*T_C)
    unsigned int Num_VNs;     // Number of Virtual Neurons (i.e., T_K*T_N*T_X_*T_Y_)
    bool folding;             // T_R x T_S x T_C < R*S*C. Neccesary to generate the signals

   
public:
    Tile(unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G,  unsigned int T_N, unsigned int T_X_, unsigned int T_Y_, bool folding); //Used by the architecture

    Tile(std::string tile_file); //Used by some external front-end to get the tile values from an input file.

    //Signals generation
    void generate_signals(int num_ms);

    //Getters
    unsigned int get_T_R()       const   {return this->T_R;}
    unsigned int get_T_S()       const   {return this->T_S;}
    unsigned int get_T_C()       const   {return this->T_C;}
    unsigned int get_T_K()       const   {return this->T_K;}
    unsigned int get_T_G()       const   {return this->T_G;}
    unsigned int get_T_N()       const   {return this->T_N;}
    unsigned int get_T_X_()      const   {return this->T_X_;}
    unsigned int get_T_Y_()      const   {return this->T_Y_;}
    unsigned int get_VN_Size()       const   {return this->VN_Size;}
    unsigned int get_Num_VNs()       const   {return this->Num_VNs;}
    

    bool get_folding_enabled()         const {return this->folding;} //Return whether this tile implies folding for the current configured network
  
    void printConfiguration(std::ofstream& out, unsigned int indent);
     

    
};




#endif
