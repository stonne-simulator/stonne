//
// Created by Francisco Munoz on 17/06/19.
//
#include "MSNetwork.h"
#include <assert.h>
#include "utility.h"
/* 
Los MS tienen una conexion forfarding con sus vecinos de la izquierda para pasarle la informacion.  Asi la informacion fluye de derecha a izquierda. Por esa razon la fuente 
del forwarding es el vecino de la derecha y el destino el de la izquierda.
*/

//By the default the three ports values will be set as one single data size
MSNetwork::MSNetwork(id_t id, std::string name, Config stonne_cfg) : MultiplierNetwork(id, name) { 
    //Extracting the input parameters
    this->ms_size = stonne_cfg.m_MSNetworkCfg.ms_size;
    this->forwarding_ports = stonne_cfg.m_MSwitchCfg.forwarding_ports;
    this->buffers_capacity = stonne_cfg.m_MSwitchCfg.buffers_capacity;
    //End of extracting the input parameters

    for(int i = 0; i < ms_size; i++) {
        std::string ms_str="MSwitch "+i;
        MSwitch* ms = new MSwitch(i, ms_str, i, stonne_cfg); //Creating the MSwitches
        this->mswitchtable[i] = ms; 
    }
    setPhysicalConnection(); //Set forwading links.
}

MSNetwork::~MSNetwork() {
    for(int i=0; i < this->ms_size; i++) {
        MSwitch* ms = this->mswitchtable[i];
        delete ms;
    }
    
    for(int i=0; i<this->ms_size-1; i++) {
        Connection* connection = this->fwconnectiontable[i]; 
        delete connection;
    }
}

//Connect a set of connections coming from the DistributionNetwork to the multipliers
void MSNetwork::setInputConnections(std::map<int, Connection*> input_connections) {
    assert(this->mswitchtable.size() == input_connections.size());
    for(std::map<int, Connection*>::iterator it=input_connections.begin(); it != input_connections.end(); ++it) {
        int index_ms = it->first; //Index value that correspond with the number of ms in the map
        Connection* conn = it->second;
        MSwitch* ms = this->mswitchtable[index_ms]; // Must exist
        ms->setInputConnection(conn);
    }

}

//Connect a set of OutputConnections coming out to the Reduction Network
void MSNetwork::setOutputConnections(std::map<int, Connection*> output_connections) {
    assert(this->mswitchtable.size() == output_connections.size());
    for(std::map<int, Connection*>::iterator it=output_connections.begin(); it != output_connections.end(); ++it) {
        int index_ms = it->first; //Index value that correspond with the number of ms in the map
        Connection* conn = it->second;
        MSwitch* ms = this->mswitchtable[index_ms]; // Must exist
        ms->setOutputConnection(conn);
    }

}


//Creating and Allocating the connections of the forwarding links
void MSNetwork::setPhysicalConnection() {
    for(int i=0; i < this->ms_size-1; i++) { //Except the last one that has no input
        Connection* connection = new Connection(this->forwarding_ports); // The ports of the connection is  a single data     
        this->fwconnectiontable[i] = connection;
        mswitchtable[i]->setInputForwardingConnection(connection); // Connection i is the input for the MS i
        mswitchtable[i+1]->setOutputForwardingConnection(connection); // Connection i is the output forwarding link from MS i+1
    }
}

std::map<int, Connection*> MSNetwork::getForwardingConnections() {
    return this->fwconnectiontable;
}

std::map<int, MSwitch*> MSNetwork::getMSwitches() {
    return this->mswitchtable;
}

//Configure each multiplier with its correpsonding virtual neuron
void MSNetwork::virtualNetworkConfig(std::map<unsigned int,unsigned int> vn_conf) {
    for(std::map<unsigned int,unsigned int>::iterator it=vn_conf.begin(); it != vn_conf.end(); ++it) {
        int index_ms = it->first; //Index value that correspond with the number of ms in the map
        int current_vn = it->second;
        MSwitch* ms = this->mswitchtable[index_ms]; // Must exist
        ms->setVirtualNeuron(current_vn);
        
    }
}

void MSNetwork::fwLinksConfig(std::map<unsigned int, bool> ms_fwsend_enabled, std::map<unsigned int, bool> ms_fwreceive_enabled) {
     //Indicating if the MS must send through the fw link
     for(std::map<unsigned int, bool>::iterator it=ms_fwsend_enabled.begin(); it != ms_fwsend_enabled.end(); ++it) {
        int index_ms = it->first; //Index value that correspond with the number of ms in the map
        bool send_signal = it->second;
        MSwitch* ms = this->mswitchtable[index_ms]; // Must exist
        ms->setOutputForwardingEnabled(send_signal);
    }

    for(std::map<unsigned int, bool>::iterator it=ms_fwreceive_enabled.begin(); it != ms_fwreceive_enabled.end(); ++it) {
        int index_ms = it->first; //Index value that correspond with the number of ms in the map
        bool receive_signal = it->second;
        MSwitch* ms = this->mswitchtable[index_ms]; // Must exist
        ms->setInputForwardingEnabled(receive_signal);
    }


}

void MSNetwork::forwardingPsumConfig(std::map<unsigned int, bool> forwarding_psum_enabled) {
    for(std::map<unsigned int, bool>::iterator it=forwarding_psum_enabled.begin(); it != forwarding_psum_enabled.end(); ++it) {
        int index_ms = it->first;
        bool forwarding_psum = it->second;
        MSwitch* ms = this->mswitchtable[index_ms];
        ms->setForwardPsum(forwarding_psum);
    }
}

void MSNetwork::directForwardingPsumConfig(std::map<unsigned int, bool> direct_forwarding_psum_enabled) {
    for(std::map<unsigned int, bool>::iterator it=direct_forwarding_psum_enabled.begin(); it != direct_forwarding_psum_enabled.end(); ++it) {
        int index_ms = it->first;
        bool direct_forwarding_psum = it->second;
        MSwitch* ms = this->mswitchtable[index_ms];
        ms->setDirectForwardPsum(direct_forwarding_psum);
    }
}

void MSNetwork::nWindowsConfig(unsigned int n_windows) {
    for(int i=0; i<this->ms_size; i++) {
        MSwitch* ms = this->mswitchtable[i];
        ms->setNWindows(n_windows); //Setting all the MSwitches with the very same value.
    }
}

void MSNetwork::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding) {
    CompilerMSN* compiler_msn = new CompilerMSN();
    compiler_msn->configureSignals(current_tile, dnn_layer, ms_size, n_folding);
    std::map<unsigned int, unsigned int> ms_vn_signals = compiler_msn->get_ms_vn_configuration();
    std::map<unsigned int, bool> ms_fwsend_enabled = compiler_msn->get_ms_fwsend_enabled();
    std::map<unsigned int, bool> ms_fwreceive_enabled = compiler_msn->get_ms_fwreceive_enabled();
    std::map<unsigned int, bool> forwarding_psum_enabled = compiler_msn->get_forwarding_psum_enabled();
    std::map<unsigned int, bool> direct_forwarding_psum_enabled = compiler_msn->get_direct_forwarding_psum_enabled();

    std::map<unsigned int, unsigned int> n_folding_configuration = compiler_msn->get_n_folding_configuration(); //Indicates forwarding multipliers. 
    this->virtualNetworkConfig(ms_vn_signals);
    this->fwLinksConfig(ms_fwsend_enabled, ms_fwreceive_enabled); //Enabling the fw links to send and receive
    this->forwardingPsumConfig(forwarding_psum_enabled);
    this->nFoldingConfig(n_folding_configuration);
    this->directForwardingPsumConfig(direct_forwarding_psum_enabled);
    unsigned int Y_=dnn_layer->get_Y_();
    this->nWindowsConfig(Y_); //N windows used to control the fw links send and receive. the number of windows in a row is Y_

    delete compiler_msn;

    
}

void MSNetwork::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size) {
    CompilerMSN* compiler_msn = new CompilerMSN();
    compiler_msn->configureSparseSignals(sparseVNs, dnn_layer, ms_size);
    std::map<unsigned int, unsigned int> ms_vn_signals = compiler_msn->get_ms_vn_configuration();
    std::map<unsigned int, bool> ms_fwsend_enabled = compiler_msn->get_ms_fwsend_enabled();
    std::map<unsigned int, bool> ms_fwreceive_enabled = compiler_msn->get_ms_fwreceive_enabled();
    std::map<unsigned int, bool> forwarding_psum_enabled = compiler_msn->get_forwarding_psum_enabled();
    std::map<unsigned int, bool> direct_forwarding_psum_enabled = compiler_msn->get_direct_forwarding_psum_enabled();
    std::map<unsigned int, unsigned int> n_folding_configuration = compiler_msn->get_n_folding_configuration();

    this->virtualNetworkConfig(ms_vn_signals);
    this->fwLinksConfig(ms_fwsend_enabled, ms_fwreceive_enabled); //Enabling the fw links to send and receive
    this->forwardingPsumConfig(forwarding_psum_enabled);
    this->directForwardingPsumConfig(direct_forwarding_psum_enabled);
    this->nFoldingConfig(n_folding_configuration);
    this->nWindowsConfig(1); //In GEMMs this is 1

    delete compiler_msn;


}

void MSNetwork::resetSignals() {
    for(int i=0; i < this->ms_size; i++) {
        MSwitch* ms = this->mswitchtable[i];
	ms->resetSignals();
    }

}


void MSNetwork::nFoldingConfig(std::map<unsigned int, unsigned int> n_folding_configuration) {
    //n_folding only is supported if there is enough buffers capacity
    unsigned int n_elements_buffers = this->buffers_capacity / sizeof(data_t);

    for(std::map<unsigned int, unsigned int>::iterator it=n_folding_configuration.begin(); it != n_folding_configuration.end(); ++it) {

        int index_ms = it->first;
        unsigned int n_folding =  it->second;
        MSwitch* ms = this->mswitchtable[index_ms];
        ms->setNFolding(n_folding); //Setting all the MSwitches with the very same value.
    }

}

void MSNetwork::cycle() {
    //Reverse order to the forwarding. The current cycle receives the data of the forwarding links sent in the previous cycle. 
    for(int i=0; i < this->ms_size; i++) {
        MSwitch* ms = mswitchtable[i];  
        ms->cycle();
    }
}

void MSNetwork::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"MSNetworkConfiguration\" : {" << std::endl;
    out << ind(indent+IND_SIZE) << "\"MSwitchConfiguration\" : [" << std::endl;  
          for(int i=0; i < this->ms_size; i++) {  //From root to leaves (without the MSs)
                MSwitch* ms = mswitchtable[i];
                ms->printConfiguration(out, indent+IND_SIZE+IND_SIZE);
                if(i==(this->ms_size-1)) {  //If I am in the last Mswitch, the comma to separate the MSwitches is not added
                    out << std::endl; //This is added because the call to ms print do not show it (to be able to put the comma, if neccesary)
                }
                else {
                    out << "," << std::endl; //Comma and line break are added to separate with the next MSwitch in the array
                }



        }
        out << ind(indent+IND_SIZE) << "]" << std::endl;

    out << ind(indent) << "}";
}

void MSNetwork::printStats(std::ofstream &out, unsigned int indent) {
    out << ind(indent) << "\"MSNetworkStats\" : {" << std::endl;
        //out << ind(indent+IND_SIZE) << "\"ms_size\" : " << this->ms_size  << std::endl; DSNetwork global parameters
        out << ind(indent+IND_SIZE) << "\"MSwitchStats\" : [" << std::endl;   //One entry per DSwitch
        for(int i=0; i < this->ms_size; i++) {  //From root to leaves (without the MSs)
                MSwitch* ms = mswitchtable[i];
                ms->printStats(out, indent+IND_SIZE+IND_SIZE);
                if(i==(this->ms_size-1)) {  //If I am in the last Mswitch, the comma to separate the MSwitches is not added
                    out << std::endl; //This is added because the call to ms print do not show it (to be able to put the comma, if neccesary)
                }
                else {
                    out << "," << std::endl; //Comma and line break are added to separate with the next MSwitch in the array
                }



        }
        out << ind(indent+IND_SIZE) << "]" << std::endl;
    out << ind(indent) << "}";

}

void MSNetwork::printEnergy(std::ofstream& out, unsigned int indent) {
    /*

      This component prints:
          - the forwarding wires
          - the mswitches counters
    */

    //Printing the forwarding wires
     for(std::map<int, Connection*>::iterator it=fwconnectiontable.begin(); it != fwconnectiontable.end(); ++it) {
         Connection* conn = fwconnectiontable[it->first];
         conn->printEnergy(out, indent, "MN_WIRE");
     }

     //Printing the mswitches counters
    
     for(std::map<int, MSwitch*>::iterator it=mswitchtable.begin(); it != mswitchtable.end(); ++it) {
         MSwitch* ms = mswitchtable[it->first];
         ms->printEnergy(out, indent);
     }


    
}
