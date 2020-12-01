#ifndef _COMPILER_ART_h_
#define _COMPILER_ART_h_

#include <iostream>
#include <string>
#include <fstream>
#include "Tile.h"
#include <assert.h>
#include "CompilerComponent.h"

/* This class configure the signals for an ANEtwork following the steps presented in MAERI Paper. */
class CompilerART : public CompilerComponent {
private:
   // Tile* current_tile;
   // unsigned int num_ms;
   // bool signals_configured;

    //Aux struct data to store the signals
    std::map<std::pair<int,int>, adderconfig_t> switches_configuration; //Adders configuration
    std::map<std::pair<int,int>, fl_t> fwlinks_configuration; //Indicates for each adder if has connection with the neighbour
    std::map<std::pair<int,int>, std::pair<bool, bool>> childs_enabled; //Indicates for each adder whether its child is enabled or not.
    std::map<std::pair<int,int>, bool> forwarding_to_memory_enabled; //Indicates for each adder whether the forwarding_to_memory link is enabled or not.

    //Aux functions
    void generate_art_signals(unsigned int num_ms); //Generate the signals for the Adder swithces
    void generate_art_enabling_links(unsigned int num_ms); //Generate the signals for the Adder switches

    //TODO. We can try to merge both sparse a non sparse. We do this now to keep the same interface
    //Aux sparse functions
    void generate_art_signals_sparse(unsigned int num_ms); //Generate the signals for the Adder swithces in sparse mode (different VN sizes)
    void generate_art_enabling_links_sparse(unsigned int num_ms); //Generate the signals for the Adder switches in sparse mode (different VN sizes)


    




public:

    CompilerART() {
        current_tile = NULL;
        signals_configured = false;
        dnn_layer=NULL;
    }
    void configureSignals(Tile* current_tile, DNNLayer* dnn_layer,  unsigned int num_ms, unsigned int n_folding); //Print the stats of the component
    void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int num_ms);
    Tile* getTile() {assert(signals_configured); return this->current_tile;}

    //Get the signals
     std::map<std::pair<int,int>, adderconfig_t> get_switches_configuration() const {return this->switches_configuration;}
    std::map<std::pair<int,int>, fl_t> get_fwlinks_configuration() const {return this->fwlinks_configuration;}
    // Indicates for each as (level, id) which one of their childs links are enabled
    // Position 0 of the pair: child left
    // Position 1 of the pair: child right
    std::map<std::pair<int,int>, std::pair<bool,bool>> get_childs_enabled() const {return this->childs_enabled;}
    std::map<std::pair<int,int>, bool> get_forwarding_to_memory_enabled() const {return this->forwarding_to_memory_enabled;}
};


#endif
