#include "SparseFlex_MSNetwork.hpp"
#include <assert.h>
#include "../../common/utility.hpp"

/*
Los MS tienen una conexion forfarding con sus vecinos de la izquierda para pasarle la informacion.  Asi la informacion fluye de derecha a izquierda. Por esa razon la fuente 
del forwarding es el vecino de la derecha y el destino el de la izquierda.
*/

//By the default the three ports values will be set as one single data size
SparseFlex_MSNetwork::SparseFlex_MSNetwork(stonne_id_t id, std::string name, Config stonne_cfg) : MultiplierNetwork(id, name) {
  //Extracting the input parameters
  this->ms_size = stonne_cfg.m_MSNetworkCfg.ms_size;
  this->forwarding_ports = stonne_cfg.m_MSwitchCfg.forwarding_ports;
  this->buffers_capacity = stonne_cfg.m_MSwitchCfg.buffers_capacity;
  //End of extracting the input parameters

  for (int i = 0; i < ms_size; i++) {
    std::string ms_str = "MSwitch " + std::to_string(i);
    SparseFlex_MSwitch* ms = new SparseFlex_MSwitch(i, ms_str, i, stonne_cfg);  //Creating the MSwitches
    this->mswitchtable[i] = ms;
  }
  setPhysicalConnection();  //Set forwading links.
}

SparseFlex_MSNetwork::~SparseFlex_MSNetwork() {
  for (int i = 0; i < this->ms_size; i++) {
    SparseFlex_MSwitch* ms = this->mswitchtable[i];
    delete ms;
  }

  for (int i = 0; i < this->ms_size - 1; i++) {
    Connection* connection = this->fwconnectiontable[i];
    delete connection;
  }
}

//Connect a set of connections coming from the DistributionNetwork to the multipliers
void SparseFlex_MSNetwork::setInputConnections(std::map<int, Connection*> input_connections) {
  assert(this->mswitchtable.size() == input_connections.size());
  for (std::map<int, Connection*>::iterator it = input_connections.begin(); it != input_connections.end(); ++it) {
    int index_ms = it->first;  //Index value that correspond with the number of ms in the map
    Connection* conn = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];  // Must exist
    ms->setInputConnection(conn);
  }
}

//Connect a set of OutputConnections coming out to the Reduction Network
void SparseFlex_MSNetwork::setOutputConnections(std::map<int, Connection*> output_connections) {
  assert(this->mswitchtable.size() == output_connections.size());
  for (std::map<int, Connection*>::iterator it = output_connections.begin(); it != output_connections.end(); ++it) {
    int index_ms = it->first;  //Index value that correspond with the number of ms in the map
    Connection* conn = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];  // Must exist
    ms->setOutputConnection(conn);
  }
}

void SparseFlex_MSNetwork::setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections) {
  std::size_t n_bus_lines = memoryConnections.size();
  std::size_t ms_per_line = this->ms_size / n_bus_lines;
  for (int i = 0; i < this->ms_size; i++) {
    SparseFlex_MSwitch* ms = this->mswitchtable[i];  // Must exist
    int busID = i / ms_per_line;
    int lineID = i % ms_per_line;
    Connection* conn = memoryConnections[busID][lineID];
    ms->setMemoryConnection(conn);
  }
}

//Creating and Allocating the connections of the forwarding links
void SparseFlex_MSNetwork::setPhysicalConnection() {
  for (int i = 0; i < this->ms_size - 1; i++) {                       //Except the last one that has no input
    Connection* connection = new Connection(this->forwarding_ports);  // The ports of the connection is  a single data
    this->fwconnectiontable[i] = connection;
    mswitchtable[i]->setInputForwardingConnection(connection);       // Connection i is the input for the MS i
    mswitchtable[i + 1]->setOutputForwardingConnection(connection);  // Connection i is the output forwarding link from MS i+1
  }
}

std::map<int, Connection*> SparseFlex_MSNetwork::getForwardingConnections() {
  return this->fwconnectiontable;
}

std::map<int, SparseFlex_MSwitch*> SparseFlex_MSNetwork::getMSwitches() {
  return this->mswitchtable;
}

//Configure each multiplier with its correpsonding virtual neuron
void SparseFlex_MSNetwork::virtualNetworkConfig(std::map<std::size_t, std::size_t> vn_conf) {
  for (std::map<std::size_t, std::size_t>::iterator it = vn_conf.begin(); it != vn_conf.end(); ++it) {
    int index_ms = it->first;  //Index value that correspond with the number of ms in the map
    int current_vn = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];  // Must exist
    ms->setVirtualNeuron(current_vn);
  }
}

void SparseFlex_MSNetwork::fwLinksConfig(std::map<std::size_t, bool> ms_fwsend_enabled, std::map<std::size_t, bool> ms_fwreceive_enabled) {
  //Indicating if the MS must send through the fw link
  for (std::map<std::size_t, bool>::iterator it = ms_fwsend_enabled.begin(); it != ms_fwsend_enabled.end(); ++it) {
    int index_ms = it->first;  //Index value that correspond with the number of ms in the map
    bool send_signal = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];  // Must exist
    ms->setOutputForwardingEnabled(send_signal);
  }

  for (std::map<std::size_t, bool>::iterator it = ms_fwreceive_enabled.begin(); it != ms_fwreceive_enabled.end(); ++it) {
    int index_ms = it->first;  //Index value that correspond with the number of ms in the map
    bool receive_signal = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];  // Must exist
    ms->setInputForwardingEnabled(receive_signal);
  }
}

void SparseFlex_MSNetwork::forwardingPsumConfig(std::map<std::size_t, bool> forwarding_psum_enabled) {
  for (std::map<std::size_t, bool>::iterator it = forwarding_psum_enabled.begin(); it != forwarding_psum_enabled.end(); ++it) {
    int index_ms = it->first;
    bool forwarding_psum = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];
    ms->setForwardPsum(forwarding_psum);
  }
}

void SparseFlex_MSNetwork::directForwardingPsumConfig(std::map<std::size_t, bool> direct_forwarding_psum_enabled) {
  for (std::map<std::size_t, bool>::iterator it = direct_forwarding_psum_enabled.begin(); it != direct_forwarding_psum_enabled.end(); ++it) {
    int index_ms = it->first;
    bool direct_forwarding_psum = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];
    ms->setDirectForwardPsum(direct_forwarding_psum);
  }
}

void SparseFlex_MSNetwork::nWindowsConfig(std::size_t n_windows) {
  for (int i = 0; i < this->ms_size; i++) {
    SparseFlex_MSwitch* ms = this->mswitchtable[i];
    ms->setNWindows(n_windows);  //Setting all the MSwitches with the very same value.
  }
}

void SparseFlex_MSNetwork::configurePartialGenerationMode(bool mergePartialSum) {
  for (int i = 0; i < this->ms_size; i++) {
    SparseFlex_MSwitch* ms = this->mswitchtable[i];
    ms->setPartialSumGenerationMode(mergePartialSum);
  }
}

void SparseFlex_MSNetwork::configureForwarderMode() {
  for (int i = 0; i < this->ms_size; i++) {
    SparseFlex_MSwitch* ms = this->mswitchtable[i];
    ms->setDirectForwardPsum(true);
  }
}

void SparseFlex_MSNetwork::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding,
                                            multiplierconfig_t multiplierconfig) {
  if (multiplierconfig == PSUM_GENERATION) {
    this->configurePartialGenerationMode(false);  //The results will go to the memory directly as they do not need to merge
  }

  else if (multiplierconfig == PSUM_GENERATION_AND_MERGE) {
    this->configurePartialGenerationMode(true);  //The results will go through the tree to be merged
  }

  else if (multiplierconfig == FORWARDER) {
    this->configureForwarderMode();
  } else {
    CompilerMSN* compiler_msn = new CompilerMSN();
    compiler_msn->configureSignals(current_tile, dnn_layer, ms_size, n_folding);
    std::map<std::size_t, std::size_t> ms_vn_signals = compiler_msn->get_ms_vn_configuration();
    std::map<std::size_t, bool> ms_fwsend_enabled = compiler_msn->get_ms_fwsend_enabled();
    std::map<std::size_t, bool> ms_fwreceive_enabled = compiler_msn->get_ms_fwreceive_enabled();
    std::map<std::size_t, bool> forwarding_psum_enabled = compiler_msn->get_forwarding_psum_enabled();
    std::map<std::size_t, bool> direct_forwarding_psum_enabled = compiler_msn->get_direct_forwarding_psum_enabled();

    std::map<std::size_t, std::size_t> n_folding_configuration = compiler_msn->get_n_folding_configuration();  //Indicates forwarding multipliers.
    this->virtualNetworkConfig(ms_vn_signals);
    this->fwLinksConfig(ms_fwsend_enabled, ms_fwreceive_enabled);  //Enabling the fw links to send and receive
    this->forwardingPsumConfig(forwarding_psum_enabled);
    this->nFoldingConfig(n_folding_configuration);
    this->directForwardingPsumConfig(direct_forwarding_psum_enabled);
    std::size_t Y_ = dnn_layer->get_Y_();
    this->nWindowsConfig(Y_);  //N windows used to control the fw links send and receive. the number of windows in a row is Y_

    delete compiler_msn;
  }
}

void SparseFlex_MSNetwork::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t ms_size) {
  CompilerMSN* compiler_msn = new CompilerMSN();
  compiler_msn->configureSparseSignals(sparseVNs, dnn_layer, ms_size);
  std::map<std::size_t, std::size_t> ms_vn_signals = compiler_msn->get_ms_vn_configuration();
  std::map<std::size_t, bool> ms_fwsend_enabled = compiler_msn->get_ms_fwsend_enabled();
  std::map<std::size_t, bool> ms_fwreceive_enabled = compiler_msn->get_ms_fwreceive_enabled();
  std::map<std::size_t, bool> forwarding_psum_enabled = compiler_msn->get_forwarding_psum_enabled();
  std::map<std::size_t, bool> direct_forwarding_psum_enabled = compiler_msn->get_direct_forwarding_psum_enabled();
  std::map<std::size_t, std::size_t> n_folding_configuration = compiler_msn->get_n_folding_configuration();

  this->virtualNetworkConfig(ms_vn_signals);
  this->fwLinksConfig(ms_fwsend_enabled, ms_fwreceive_enabled);  //Enabling the fw links to send and receive
  this->forwardingPsumConfig(forwarding_psum_enabled);
  this->directForwardingPsumConfig(direct_forwarding_psum_enabled);
  this->nFoldingConfig(n_folding_configuration);
  this->nWindowsConfig(1);  //In GEMMs this is 1

  delete compiler_msn;
}

void SparseFlex_MSNetwork::resetSignals() {
  for (int i = 0; i < this->ms_size; i++) {
    SparseFlex_MSwitch* ms = this->mswitchtable[i];
    ms->resetSignals();
  }
}

void SparseFlex_MSNetwork::nFoldingConfig(std::map<std::size_t, std::size_t> n_folding_configuration) {
  //n_folding only is supported if there is enough buffers capacity
  std::size_t n_elements_buffers = this->buffers_capacity / sizeof(data_t);

  for (std::map<std::size_t, std::size_t>::iterator it = n_folding_configuration.begin(); it != n_folding_configuration.end(); ++it) {

    int index_ms = it->first;
    std::size_t n_folding = it->second;
    SparseFlex_MSwitch* ms = this->mswitchtable[index_ms];
    ms->setNFolding(n_folding);  //Setting all the MSwitches with the very same value.
  }
}

void SparseFlex_MSNetwork::cycle() {
  //Reverse order to the forwarding. The current cycle receives the data of the forwarding links sent in the previous cycle.
  for (int i = 0; i < this->ms_size; i++) {
    SparseFlex_MSwitch* ms = mswitchtable[i];
    ms->cycle();
  }
}

void SparseFlex_MSNetwork::printConfiguration(std::ofstream& out, std::size_t indent) {
  out << ind(indent) << "\"SparseFlex_MSNetworkConfiguration\" : {" << std::endl;
  out << ind(indent + IND_SIZE) << "\"MSwitchConfiguration\" : [" << std::endl;
  for (int i = 0; i < this->ms_size; i++) {  //From root to leaves (without the MSs)
    SparseFlex_MSwitch* ms = mswitchtable[i];
    ms->printConfiguration(out, indent + IND_SIZE + IND_SIZE);
    if (i == (this->ms_size - 1)) {  //If I am in the last Mswitch, the comma to separate the MSwitches is not added
      out << std::endl;              //This is added because the call to ms print do not show it (to be able to put the comma, if neccesary)
    } else {
      out << "," << std::endl;  //Comma and line break are added to separate with the next MSwitch in the array
    }
  }
  out << ind(indent + IND_SIZE) << "]" << std::endl;

  out << ind(indent) << "}";
}

void SparseFlex_MSNetwork::printStats(std::ofstream& out, std::size_t indent) {
  out << ind(indent) << "\"SparseFlex_MSNetworkStats\" : {" << std::endl;
  //out << ind(indent+IND_SIZE) << "\"ms_size\" : " << this->ms_size  << std::endl; DSNetwork global parameters
  out << ind(indent + IND_SIZE) << "\"MSwitchStats\" : [" << std::endl;  //One entry per DSwitch
  for (int i = 0; i < this->ms_size; i++) {                              //From root to leaves (without the MSs)
    SparseFlex_MSwitch* ms = mswitchtable[i];
    ms->printStats(out, indent + IND_SIZE + IND_SIZE);
    if (i == (this->ms_size - 1)) {  //If I am in the last Mswitch, the comma to separate the MSwitches is not added
      out << std::endl;              //This is added because the call to ms print do not show it (to be able to put the comma, if neccesary)
    } else {
      out << "," << std::endl;  //Comma and line break are added to separate with the next MSwitch in the array
    }
  }
  out << ind(indent + IND_SIZE) << "]" << std::endl;
  out << ind(indent) << "}";
}

void SparseFlex_MSNetwork::printEnergy(std::ofstream& out, std::size_t indent) {
  /*

      This component prints:
          - the forwarding wires
          - the mswitches counters
    */

  //Printing the forwarding wires
  for (std::map<int, Connection*>::iterator it = fwconnectiontable.begin(); it != fwconnectiontable.end(); ++it) {
    Connection* conn = fwconnectiontable[it->first];
    conn->printEnergy(out, indent, "MN_WIRE");
  }

  //Printing the mswitches counters

  for (std::map<int, SparseFlex_MSwitch*>::iterator it = mswitchtable.begin(); it != mswitchtable.end(); ++it) {
    SparseFlex_MSwitch* ms = mswitchtable[it->first];
    ms->printEnergy(out, indent);
  }
}

/*MSNetworkStats SparseFlex_MSNetwork::getStats() {
    MSNetworkStats msnetworkStats;
    //Collecting MSwitches stats
    for(std::map<int, SparseFlex_MSwitch*>::iterator it=mswitchtable.begin(); it != mswitchtable.end(); ++it) {
         SparseFlex_MSwitch* ms = mswitchtable[it->first];
	 MSwitchStats mswitchStats = ms->getStats();
	 FifoStats fifo_activation_stats = ms->getActivationFifo()->getStats();
	 FifoStats fifo_weight_stats = ms->getWeightFifo()->getStats();
	 FifoStats fifo_psum_stats = ms->getPsumFifo()->getStats();
         msnetworkStats.n_multiplications+=mswitchStats.n_multiplications;
         msnetworkStats.n_l1_weight_reads+=(fifo_weight_stats.n_pops+fifo_weight_stats.n_fronts);
         msnetworkStats.n_l1_weight_writes+=mswitchStats.n_weights_receive;
         msnetworkStats.n_l1_input_writes+=mswitchStats.n_inputs_receive;
         msnetworkStats.n_l1_input_reads+=(fifo_activation_stats.n_pops+fifo_activation_stats.n_fronts);
         msnetworkStats.n_l1_psum_writes+=mswitchStats.n_psums_receive;
         msnetworkStats.n_l1_psum_reads+=(fifo_psum_stats.n_pops+fifo_psum_stats.n_fronts);
 
    }

    //Collecting connection stats
    for(std::map<int, Connection*>::iterator it=fwconnectiontable.begin(); it != fwconnectiontable.end(); ++it) {
         Connection* conn = fwconnectiontable[it->first];
	 ConnectionStats conn_stats = conn->getStats();
	 msnetworkStats.n_local_network_traversals+=conn_stats.n_sends;
     }

    return msnetworkStats;
}*/
