#ifndef _COMPILER_MULTIPLIERMESH_h_
#define _COMPILER_MULTIPLIERMESH_h_

#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include "common/dnn/Tile.hpp"
#include "CompilerComponent.hpp"

class CompilerMultiplierMesh : public CompilerComponent {
 private:
  //Multiplier signals
  std::map<std::pair<int, int>, bool> forwarding_bottom_enabled;  //Forwarding to the bottom MS
  std::map<std::pair<int, int>, bool> forwarding_right_enabled;   //Forwrding to the right ms
  std::map<std::pair<int, int>, std::size_t> ms_vn_configuration;
  std::size_t ms_rows;
  std::size_t ms_cols;

  //Aux functions
  void generate_ms_signals(std::size_t ms_rows, std::size_t ms_cols);  //The function in charge to generate the signals for the Multiplier

 public:
  CompilerMultiplierMesh() {
    current_tile = NULL;
    signals_configured = false;
    dnn_layer = NULL;
  }

  void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_rows, std::size_t ms_cols);
  void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t num_ms);

  Tile* getTile() {
    assert(signals_configured);
    return this->current_tile;
  }

  //Get the signals
  std::map<std::pair<int, int>, std::size_t> get_ms_vn_configuration() const { return this->ms_vn_configuration; }

  std::map<std::pair<int, int>, bool> get_forwarding_bottom_enabled() const { return this->forwarding_bottom_enabled; }

  std::map<std::pair<int, int>, bool> get_forwarding_right_enabled() const { return this->forwarding_right_enabled; }
};

#endif
