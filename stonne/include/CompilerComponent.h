#ifndef _COMPILER_COMPONENT_h_
#define _COMPILER_COMPONENT_h_

#include <iostream>
#include <string>
#include <fstream>
#include "Tile.h"
#include "DNNLayer.h"
#include <assert.h>
#include <vector>

class CompilerComponent {


public:
    Tile* current_tile;
    std::vector<SparseVN> sparseVNs;
    DNNLayer* dnn_layer;
    unsigned int num_ms;
    bool signals_configured;
    unsigned int n_folding;
    
    CompilerComponent() {
        current_tile = NULL;
        signals_configured = false;
        this->dnn_layer=NULL;
    }
    virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int num_ms, unsigned int n_folding) {} //Print the stats of the component
    virtual void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int num_ms) {}
    Tile* getTile() {assert(signals_configured); return this->current_tile;}
};


#endif
