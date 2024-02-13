#ifndef __TILE__H
#define __TILE__H

#include <iostream>
#include <map>
#include "types.hpp"

/*
 * This class represents a sparse cluster mapped onto the architecture. 
 * Basically, size is the number of multipliers needed, and folding indicates if this cluster
 * requires an  extra multiplier to act as a forwarder.
 */
class SparseVN {
 private:
  std::size_t size;
  bool folding;

 public:
  SparseVN(std::size_t size, bool folding) {
    this->size = size;
    this->folding = folding;
  }

  std::size_t get_VN_Size() {
    if (this->folding) {
      return this->size + 1;
    } else {
      return this->size;
    }
  }

  bool getFolding() { return this->folding; }
};

/*
 This class represent a tile 
*/

class Tile {
 private:
  std::size_t T_R;      // Number of filter rows
  std::size_t T_S;      // Number of filter columns
  std::size_t T_C;      // Number of input and filter channels
  std::size_t T_K;      // Number of filters and number of ofmap channels per group
  std::size_t T_G;      // Number of groups
  std::size_t T_N;      // Batch size
  std::size_t T_X_;     // Number of output fmap rows
  std::size_t T_Y_;     // Number of output fmap columns
  std::size_t VN_Size;  // Virtual Neuron Size (i.e., T_R*T_S*T_C)
  std::size_t Num_VNs;  // Number of Virtual Neurons (i.e., T_K*T_N*T_X_*T_Y_)
  bool folding;         // T_R x T_S x T_C < R*S*C. Neccesary to generate the signals

 public:
  Tile(std::size_t T_R, std::size_t T_S, std::size_t T_C, std::size_t T_K, std::size_t T_G, std::size_t T_N, std::size_t T_X_, std::size_t T_Y_,
       bool folding);  //Used by the architecture

  Tile(std::string tile_file);  //Used by some external front-end to get the tile values from an input file.

  //Signals generation
  void generate_signals(int num_ms);

  //Getters
  std::size_t get_T_R() const { return this->T_R; }

  std::size_t get_T_S() const { return this->T_S; }

  std::size_t get_T_C() const { return this->T_C; }

  std::size_t get_T_K() const { return this->T_K; }

  std::size_t get_T_G() const { return this->T_G; }

  std::size_t get_T_N() const { return this->T_N; }

  std::size_t get_T_X_() const { return this->T_X_; }

  std::size_t get_T_Y_() const { return this->T_Y_; }

  std::size_t get_VN_Size() const { return this->VN_Size; }

  std::size_t get_Num_VNs() const { return this->Num_VNs; }

  bool get_folding_enabled() const { return this->folding; }  //Return whether this tile implies folding for the current configured network

  void printConfiguration(std::ofstream& out, std::size_t indent);
};

#endif
