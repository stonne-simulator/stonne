#include "CompilerMSN.h"
#include "Tile.h"
#include "utility.h"
#include <math.h>
#include "types.h"
#include <assert.h>
#include "cpptoml.h"

void CompilerMSN::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int num_ms, unsigned int n_folding) {
    assert(ispowerof2(num_ms));
    assert(current_tile->get_VN_Size()*current_tile->get_Num_VNs() <= num_ms);
    this->current_tile = current_tile;
    this->dnn_layer = dnn_layer;
    this->num_ms = num_ms;
    this->n_folding=n_folding;
    this->signals_configured = true;
    //Configuring Multiplier switches
    this->generate_ms_signals(num_ms);
    
}

void CompilerMSN::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int num_ms) {
    assert(ispowerof2(num_ms));
    //Checking if there are enough multipliers
    int num_ms_used = 0;
    for(int i=0; i<sparseVNs.size(); i++) {
        num_ms_used+=sparseVNs[i].get_VN_Size(); 
    }

    assert(num_ms_used<=num_ms);
    this->sparseVNs = sparseVNs;
    this->dnn_layer = dnn_layer;
    this->num_ms = num_ms;
    this->signals_configured = true;
    //Configuring Multiplier switches
    this->generate_ms_sparse_signals(num_ms);

}


void CompilerMSN::generate_ms_signals(unsigned int num_ms) {
    //1. Indicating to each MS its corresponding VN ID
    //Saving the number of iterations needed. Used in the cases where an extra MS is needed
    //unsigned int n_folding=(this->dnn_layer->get_R() / this->current_tile->get_T_R())*(this->dnn_layer->get_S() / this->current_tile->get_T_S()) * (this->dnn_layer->get_C() / this->current_tile->get_T_C());
    for(int i=0; i < this->current_tile->get_Num_VNs(); i++) {
        for(int j=0; j < this->current_tile->get_VN_Size(); j++) {
            unsigned int ms_index = i*this->current_tile->get_VN_Size() + j;
            ms_vn_configuration[ms_index]=i; //Allocating the corresponding VN.
	    n_folding_configuration[ms_index]=n_folding; //Allocating the number of folds. In dense all the same
	    direct_forwarding_psum_enabled[ms_index]=false;

            if(this->current_tile->get_folding_enabled() && (j==0)) { //The first MS of each VN is the aux MS to accumulate psums
                forwarding_psum_enabled[ms_index]=true;
            }
     
            else {
                forwarding_psum_enabled[ms_index]=false;
            }
            
           
        }
    }

    //2. Indicating to each MS wether it has to receive or send data

    if(this->current_tile->get_T_Y_() > 1) {  //Conditions to enable the fw links. T_Y_ must be greater than 1 and stride must be 1. If not, all the signals must be false
        for(int i=0; i < this->current_tile->get_Num_VNs(); i++) {
            for(int j=0; j < this->current_tile->get_VN_Size(); j++) {
                unsigned int ms_index = i*this->current_tile->get_VN_Size() + j;
                ms_fwsend_enabled[ms_index]=false;
                ms_fwreceive_enabled[ms_index]=false; 
         
            }
        }

    }

    else  {
        unsigned int shift_ms = 0;
        if(this->current_tile->get_folding_enabled()) {
            shift_ms =1; //If there is folding we leave a ms to accumulate 
        }
        for(int i=0; i < this->current_tile->get_Num_VNs(); i++) {
            for(int c=0; c < this->current_tile->get_T_C(); c++) {  //For each channel
                for(int r=0; r < this->current_tile->get_T_R() ; r++) {   //For each row 
                    for(int s=0; s < this->current_tile->get_T_S(); s++) {
                        int ms_index = i*this->current_tile->get_VN_Size() + c*this->current_tile->get_T_R()*this->current_tile->get_T_S() + r*this->current_tile->get_T_S() + s + shift_ms; //Note that we sum shift_ms
                        // Indicating to each MS whether it has to send data to the fw link (MS LEFT) or not .
                        if(s > 0) { //If the ms does not contain one first column, it has to send to the left
                            ms_fwsend_enabled[ms_index]=true;
                        }
                        else {
                            ms_fwsend_enabled[ms_index]=false;
                        }

                        // Indicating to each MS whether it has to receive data from the fw link (MS RIGHT) or not.
                        if(s < (this->current_tile->get_T_S()-1)) { //If the ms does not map one last column, it has to receive data from the right
                            ms_fwreceive_enabled[ms_index]=true;
                        }

                        else {
                            ms_fwreceive_enabled[ms_index]=false;
                        }

                    
                    }
                }
            }
        
        } //End for i

    } //End else     

    //Disabling if padding
     //If stride > 1 then all the signals of ms_fwreceive_enabled and ms_fwsend_enabled must be disabled since no reuse between MSwitches can be done. In order to not to incorporate stride
    //as a tile parameter, we leave the class Tile not aware of the stride. Then, if stride exists, here the possible enabled signals (since tile does not know about tile) are disabled.
    if(this->dnn_layer->get_strides() > 1) {
        for(unsigned int i=0; i<num_ms; i++) {
            ms_fwsend_enabled[i]=false;
            ms_fwreceive_enabled[i]=false;
        }
    }



}

//Note that since this is the sparse configuration, this is to run GEMMs, and therefore the forwarding links among MSs are
//always disabled
void CompilerMSN::generate_ms_sparse_signals(unsigned int num_ms) {
    //1. Indicating to each MS its corresponding VN ID
    //Saving the number of iterations needed. Used in the cases where an extra MS is needed
    unsigned int ms_index = 0;
    for(int i=0; i < this->sparseVNs.size(); i++) {
        for(int j=0; j < this->sparseVNs[i].get_VN_Size(); j++) {
            ms_vn_configuration[ms_index]=i; //Allocating the corresponding VN.
	    n_folding_configuration[ms_index]=1; //TODO change this later
	    forwarding_psum_enabled[ms_index]=false; //In sparse this type of forwrding is always false
            if(this->sparseVNs[i].getFolding() && (j==0)) { //The first MS of each VN is the aux MS to accumulate psums
                direct_forwarding_psum_enabled[ms_index]=true;
            }
     
            else {
                direct_forwarding_psum_enabled[ms_index]=false;
            }
            
	      //2. Indicating to each MS wether it has to receive or send data. In this case, since this is to run GEMM, 
	      //the fw links must be disabled

	    ms_fwsend_enabled[ms_index]=false;
            ms_fwreceive_enabled[ms_index]=false;


	    ms_index++;
           
           
        }


    }

  
}




