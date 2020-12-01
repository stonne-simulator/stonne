#include "CompilerMultiplierMesh.h"
#include "Tile.h"
#include "utility.h"
#include <math.h>
#include "types.h"
#include <assert.h>
#include "cpptoml.h"

void CompilerMultiplierMesh::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_rows, unsigned int ms_cols) {
    assert(current_tile->get_T_K() <= ms_cols); //Number of filters
    assert(current_tile->get_T_X_()*current_tile->get_T_Y_() <= ms_rows); //Number of conv windows
    this->current_tile = current_tile;
    this->dnn_layer = dnn_layer;
    this->ms_rows = ms_rows;
    this->ms_cols = ms_cols;
    this->signals_configured = true;
    //Configuring Multiplier switches
    this->generate_ms_signals(ms_rows, ms_cols);
    
}

void CompilerMultiplierMesh::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int num_ms) {
    assert(false); //TPU implementation does not allow sprsity due to its rigit nature
}


void CompilerMultiplierMesh::generate_ms_signals(unsigned int ms_rows, unsigned int ms_cols) {
    unsigned int rows_used = this->current_tile->get_T_X_()*this->current_tile->get_T_Y_();
    unsigned int cols_used = this->current_tile->get_T_K();
    //Bottom and right signals 
    for(int i=0; i<ms_rows; i++) {
        for(int j=0; j<ms_cols; j++) {
	    std::pair<int,int> ms_index(i,j);

	    if((i < rows_used) && (j < cols_used)) {
                 unsigned int VN = i*cols_used+j;
                 ms_vn_configuration[ms_index]=VN;
	    }

            if((i < (rows_used-1)) && (j < cols_used)) {
                forwarding_bottom_enabled[ms_index]=true;
	    }

	    else {
                forwarding_bottom_enabled[ms_index]=false;
	    }

	    if((j < (cols_used-1)) && (i < rows_used)) {
                forwarding_right_enabled[ms_index]=true;
	    }

	    else {
                forwarding_right_enabled[ms_index]=false;
	    }
	}
    
    }
              
}
