#ifndef _COMPILER_MSN_h_
#define _COMPILER_MSN_h_

#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include "CompilerComponent.hpp"
#include "Tile.hpp"

/* This class configure the signals for an ANEtwork following the steps presented in MAERI Paper. */
class CompilerMSN : public CompilerComponent {
 private:
  //Aux struct data to store the signals
  //Multiplier signals
  std::map<std::size_t, std::size_t> ms_vn_configuration;      //Virtual neuron of each MS configuration
  std::map<std::size_t, bool> ms_fwsend_enabled;               //Indicates for each MS if must send data to the fw link (MS LEFT)
  std::map<std::size_t, bool> ms_fwreceive_enabled;            //Indicates for each MS if must receive data from the fw link (MS RIGHT)
  std::map<std::size_t, bool> forwarding_psum_enabled;         //Indicates if the MS has to forward psums or otherwise has to act as a normal multiplier.
  std::map<std::size_t, bool> direct_forwarding_psum_enabled;  //Indicates if the MS has to forward psums WITHOUT any control.
  std::map<std::size_t, std::size_t> n_folding_configuration;  //Indicates the number of folds that each MS is going to perform

  //Aux functions
  void generate_ms_signals(std::size_t num_ms);         //The function in charge to generate the signals for the MSwitches
  void generate_ms_sparse_signals(std::size_t num_ms);  //Generate signals for the MSwitches taking into account the different size clusters.

 public:
  CompilerMSN() {
    current_tile = NULL;
    signals_configured = false;
    dnn_layer = NULL;
  }

  void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t num_ms, std::size_t n_folging);
  void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t num_ms);

  Tile* getTile() {
    assert(signals_configured);
    return this->current_tile;
  }

  //Get the signals
  std::map<std::size_t, std::size_t> get_ms_vn_configuration() const { return this->ms_vn_configuration; }

  std::map<std::size_t, bool> get_ms_fwsend_enabled() const { return this->ms_fwsend_enabled; }

  std::map<std::size_t, bool> get_ms_fwreceive_enabled() const { return this->ms_fwreceive_enabled; }

  std::map<std::size_t, bool> get_forwarding_psum_enabled() const { return this->forwarding_psum_enabled; }

  std::map<std::size_t, bool> get_direct_forwarding_psum_enabled() const { return this->direct_forwarding_psum_enabled; }

  std::map<std::size_t, std::size_t> get_n_folding_configuration() const { return this->n_folding_configuration; }
};

#endif
