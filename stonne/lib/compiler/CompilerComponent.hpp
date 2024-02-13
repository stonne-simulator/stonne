#ifndef _COMPILER_COMPONENT_h_
#define _COMPILER_COMPONENT_h_

#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../common/dnn/DNNLayer.hpp"
#include "../common/dnn/Tile.hpp"

class CompilerComponent {

 public:
  Tile* current_tile;
  std::vector<SparseVN> sparseVNs;
  DNNLayer* dnn_layer;
  std::size_t num_ms;
  bool signals_configured;
  std::size_t n_folding;

  CompilerComponent() {
    current_tile = NULL;
    signals_configured = false;
    this->dnn_layer = NULL;
  }

  virtual ~CompilerComponent() {}

  virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t num_ms, std::size_t n_folding) = 0;  //Print the stats of the component

  virtual void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t num_ms) = 0;

  Tile* getTile() {
    assert(signals_configured);
    return this->current_tile;
  }
};

#endif
