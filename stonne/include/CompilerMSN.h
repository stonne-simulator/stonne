#ifndef _COMPILER_MSN_h_
#define _COMPILER_MSN_h_

#include <iostream>
#include <string>
#include <fstream>
#include "Tile.h"
#include <assert.h>
#include "CompilerComponent.h"

/* This class configure the signals for an ANEtwork following the steps presented in MAERI Paper. */
class CompilerMSN : public CompilerComponent{
private:

    //Aux struct data to store the signals
      //Multiplier signals
    std::map<unsigned int, unsigned int> ms_vn_configuration; //Virtual neuron of each MS configuration
    std::map<unsigned int, bool> ms_fwsend_enabled; //Indicates for each MS if must send data to the fw link (MS LEFT)
    std::map<unsigned int, bool> ms_fwreceive_enabled; //Indicates for each MS if must receive data from the fw link (MS RIGHT)
    std::map<unsigned int, bool> forwarding_psum_enabled; //Indicates if the MS has to forward psums or otherwise has to act as a normal multiplier. 
    std::map<unsigned int, bool> direct_forwarding_psum_enabled; //Indicates if the MS has to forward psums WITHOUT any control.
    std::map<unsigned int, unsigned int> n_folding_configuration; //Indicates the number of folds that each MS is going to perform

    //Aux functions
    void generate_ms_signals(unsigned int num_ms); //The function in charge to generate the signals for the MSwitches
    void generate_ms_sparse_signals(unsigned int num_ms); //Generate signals for the MSwitches taking into account the different size clusters.


    




public:

    CompilerMSN() {
        current_tile = NULL;
        signals_configured = false;
        dnn_layer=NULL;
    }
    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int num_ms, unsigned int n_folging) ; 
    void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int num_ms);
    Tile* getTile() {assert(signals_configured); return this->current_tile;}

    //Get the signals
    std::map<unsigned int, unsigned int> get_ms_vn_configuration() const {return this->ms_vn_configuration;}
    std::map<unsigned int, bool> get_ms_fwsend_enabled() const {return this->ms_fwsend_enabled;}
    std::map<unsigned int, bool> get_ms_fwreceive_enabled() const {return this->ms_fwreceive_enabled;}
    std::map<unsigned int, bool> get_forwarding_psum_enabled() const {return this->forwarding_psum_enabled;}
    std::map<unsigned int, bool> get_direct_forwarding_psum_enabled() const {return this->direct_forwarding_psum_enabled;}
    std::map<unsigned int, unsigned int> get_n_folding_configuration() const {return this->n_folding_configuration;} 

};


#endif
