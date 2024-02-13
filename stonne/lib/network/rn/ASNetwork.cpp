#include "ASNetwork.hpp"
#include <assert.h>
#include <math.h>
#include "../../common/utility.hpp"

//TODO Conectar los enlaces intermedios de forwarding
//This Constructor creates the reduction tree similar to the one shown in the paper
ASNetwork::ASNetwork(stonne_id_t id, std::string name, Config stonne_cfg, Connection* outputConnection) : ReduceNetwork(id, name) {
  // Collecting the parameters from configuration file
  this->m_stonneCfg = stonne_cfg;
  this->m_portWidth = stonne_cfg.m_ASwitchCfg.port_width;
  this->m_msSize = stonne_cfg.m_MSNetworkCfg.ms_size;
  this->m_accumulationBufferEnabled = stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled;
  //End collecting the parameters from the configuration file
  assert(ispowerof2(m_msSize));  //Ensure the number of multipliers is power of 2.
  this->p_outputConnection = outputConnection;
  int nlevels = log10(m_msSize) / log10(2);  //All the levels without count the leaves (MSwitches)
  this->m_nlevels = nlevels;
  int adders_this_level = 1;
  std::size_t as_id = 0;
  for (int i = 0; i < this->m_nlevels; i++) {      //From root to leaves (without the MSs)
    for (int j = 0; j < adders_this_level; j++) {  // From left to right of the structure
      std::string as_str = "ASwitch " + std::to_string(as_id);
      ASwitch* as = new ASwitch(as_id, as_str, i, j, stonne_cfg);
      as_id += 1;  //increasing the as id
      std::pair<int, int> levelandnum(i, j);
      m_aswitchtable[levelandnum] = as;

      /* Now its the bus who create this
            //Connecting with output memory. 
            //Creating the connection between this adder and the collectionBus
            Connection* memoryConnection = new Connection(output_ports);
            as->setMemoryConnection(memoryConnection);
            memoryconnectiontable[levelandnum]=memoryConnection;
            */

      if (i == 0) {  //The first node is connected to the outputConnection.
        as->setOutputConnection(outputConnection);
        this->m_singleSwitches.push_back(as);  //The first node is a single reduction switch since it has no forwarding links.
      }

      else {  //Connecting level i nodes with level -1 connections that are the output connections of the previous level. THere are the same number of output connections and nodes.
        std::pair<int, int> sourcepair(i - 1, j);  // the same number of connections in the above level matches the number of Dswitches in this l
        Connection* correspondingOutputConnection =
            m_inputconnectiontable[sourcepair];  //output node i is the input of the node -i, already connection in the previous loop (code below)
        as->setOutputConnection(correspondingOutputConnection);

        //Connecting forwarding connections (intermedium links). Even (par) node with level>0 and node>0) is the second second node of a node with fw connection.
        if (j && ((j % 2) == 0)) {  //FW link is needed. Connecting the consecutive node that does not share parent with the previous one
          //Creating the fw link
          Connection* fwconnection = new Connection(m_portWidth);
          //Adding the fw to the map. The index is the second node of the pair
          std::pair<int, int> fwpair(i, j);
          m_forwardingconnectiontable[fwpair] = fwconnection;
          //Connecting fw link to the current one and the previous node, which has no shared parent
          as->setForwardingConnection(fwconnection);
          std::pair<int, int> previous_pair(i, j - 1);
          ASwitch* previous_as = m_aswitchtable[previous_pair];
          previous_as->setForwardingConnection(fwconnection);  //Connected with the same fwlink
          //Inserting to the list double reduction switch. We insert both as double but actually both are an unit
          this->m_doubleSwitches.push_back(previous_as);
          this->m_doubleSwitches.push_back(as);

        }

        else if ((j == 0) || (j == (adders_this_level - 1))) {  //If it is the first or the last as then is a single reduction switch
          this->m_singleSwitches.push_back(as);
        }
      }

      //Creating and Connecting input connections (left and right)
      //For each connection of this particular node
      for (int c = 0; c < CONNECTIONS_PER_SWITCH; c++) {
        Connection* connection = new Connection(m_portWidth);            //Output link so output ports
        int connection_pos_this_level = j * CONNECTIONS_PER_SWITCH + c;  //number of switches alreay created + shift this switch
        std::pair<int, int> connectionpair(i, connection_pos_this_level);
        m_inputconnectiontable[connectionpair] = connection;
        //Connecting adder with its input connections
        if (c == LEFT) {
          as->setInputLeftConnection(connection);
        }

        else if (c == RIGHT) {
          as->setInputRightConnection(connection);
        }
      }
    }
    //In the next level, the input ports is the output ports of this level
    adders_this_level = adders_this_level * 2;
  }
}

ASNetwork::~ASNetwork() {
  //Delete the adders from aswitchtable
  for (std::map<std::pair<int, int>, ASwitch*>::iterator it = m_aswitchtable.begin(); it != m_aswitchtable.end(); ++it) {
    delete it->second;
  }

  //Removing connections from inputconnectiontable
  for (std::map<std::pair<int, int>, Connection*>::iterator it = m_inputconnectiontable.begin(); it != m_inputconnectiontable.end(); ++it) {
    delete it->second;
  }

  //Delete forwarding connections
  for (std::map<std::pair<int, int>, Connection*>::iterator it = m_forwardingconnectiontable.begin(); it != m_forwardingconnectiontable.end(); ++it) {
    delete it->second;
  }

  if (this->m_accumulationBufferEnabled > 0) {
    delete this->p_accumulationBuffer;
    for (int i = 0; i < m_accumulationbufferconnectiontable.size(); i++) {
      delete this->m_accumulationbufferconnectiontable[i];
    }
  }
}

void ASNetwork::setMemoryConnections(std::vector<std::vector<Connection*>> memoryConnections) {
  std::size_t n_bus_lines = memoryConnections.size();

  //If the accumulation buffer is enabled then the object must be created
  if (this->m_accumulationBufferEnabled > 0) {
    this->p_accumulationBuffer = new AccumulationBuffer(0, "AccumulationBuffer", this->m_stonneCfg, m_aswitchtable.size());
  }
  // Interconnect double reduction switches
  // Just in case accumulation buffer exists
  std::size_t connectionID = 0;
  std::vector<Connection*> accbuffer_memory_connections;
  for (int i = 0; i < m_doubleSwitches.size(); i += 2) {  //2 by 2 since each 2 aswitches form a single double switch
    std::size_t inputID = 2 * ((i / 2) / n_bus_lines);
    std::size_t busID = (i / 2) % n_bus_lines;
    ASwitch* as_first = m_doubleSwitches[i];
    ASwitch* as_second = m_doubleSwitches[i + 1];
    assert(busID < memoryConnections.size());                        //Making sure the CollectionBus returns the correct busLine
    assert((inputID + 1) < memoryConnections[busID].size());         //Making sure the CollectionBus returns the correct busLine
    Connection* mem_conn_first = memoryConnections[busID][inputID];  //Connecting as i to busID, inputID connection
    Connection* mem_conn_second = memoryConnections[busID][inputID + 1];
    if (this->m_accumulationBufferEnabled == 0) {  //If the accumulation buffer is not enabled
      as_first->setMemoryConnection(mem_conn_first, busID, inputID);
      as_second->setMemoryConnection(mem_conn_second, busID, inputID + 1);
    }

    else {  //making the connection with the accumulation buffer as well
      Connection* accbufferconnection_first = new Connection(m_portWidth);
      Connection* accbufferconnection_second = new Connection(m_portWidth);
      m_accumulationbufferconnectiontable.push_back(accbufferconnection_first);
      m_accumulationbufferconnectiontable.push_back(accbufferconnection_second);
      as_first->setMemoryConnection(accbufferconnection_first, busID, inputID);  //Connecting with the acc buffer
      as_second->setMemoryConnection(accbufferconnection_second, busID, inputID + 1);
      accbuffer_memory_connections.push_back(mem_conn_first);
      accbuffer_memory_connections.push_back(mem_conn_second);
    }
    connectionID += 2;
  }

  for (int i = 0; i < m_singleSwitches.size(); i++) {
    std::size_t inputID_Base = (m_doubleSwitches.size() / n_bus_lines) +
                               1;  //Noice that here we do not use 2*double_switches since there are single switches in the list considered as double
    std::size_t inputID = inputID_Base + (i / n_bus_lines);
    std::size_t busID = (i + m_doubleSwitches.size()) % n_bus_lines;
    ASwitch* as = m_singleSwitches[i];
    Connection* mem_conn = memoryConnections[busID][inputID];
    if (this->m_accumulationBufferEnabled == 0) {  //If the accumulation buffer is not enabled
      as->setMemoryConnection(mem_conn, busID, inputID);
    }

    else {
      Connection* accbufferconnection = new Connection(m_portWidth);
      m_accumulationbufferconnectiontable.push_back(accbufferconnection);
      as->setMemoryConnection(accbufferconnection, busID, inputID);  //Connecting with the acc buffer
      accbuffer_memory_connections.push_back(mem_conn);
    }

    connectionID += 1;
  }

  //Finally we connect the links with the accumulation buffer if exists
  if (this->m_accumulationBufferEnabled > 0) {
    this->p_accumulationBuffer->setInputConnections(m_accumulationbufferconnectiontable);
    this->p_accumulationBuffer->setMemoryConnections(accbuffer_memory_connections);
  }
}

std::map<int, Connection*> ASNetwork::getLastLevelConnections() {
  int last_level_index = this->m_nlevels - 1;       //The levels start from 0, so in the table the last level is nlevels-1
  std::map<int, Connection*> connectionsLastLevel;  //Map with the connections of the last level of AS (the ones that must be cnnected with the MN)
  for (int i = 0; i < this->m_msSize; i++) {        //Each multiplier must have its own connection if the ASNetwork has been created correctly
    std::pair<int, int> current_connection(last_level_index, i);
    connectionsLastLevel[i] = this->m_inputconnectiontable[current_connection];
  }
  return connectionsLastLevel;
}

/*
    This function set a different configuration for each switch of the network. Each pair i, j corresponds to the level and the id in the
    level for each switch. Each pair has a correspondency with a configuration (ADD_2_1, ADD_3_1, ..., etc.)
*/
void ASNetwork::addersConfiguration(std::map<std::pair<int, int>, adderconfig_t> adder_configurations) {
  /* This code is for iterating over all the switches. Forcing the user to set a configuration for each switch (when maybe there is some switch that does not have to)
    int switches_this_level = 1;
    for(int i=0; i< nlevels; i++) {
        for(int j=0; j < switches_this_level; j++) {
            std::pair<int, int> current_switch_pair(i,j);
            ASwitch* as = aswitchtable[current_switch_pair];
            adderconfig_t conf = adder_configurations[current_switch_pair]; //Getting the configuration for that specific adder
            as->setConfigurationMode(conf);
            //Setting

            
        }
        switches_this_level=switches_this_level*2;
    }  
   */
  for (std::map<std::pair<int, int>, adderconfig_t>::iterator it = adder_configurations.begin(); it != adder_configurations.end(); ++it) {
    adderconfig_t conf = it->second;
    ASwitch* as = m_aswitchtable[it->first];  //pair index
    as->setConfigurationMode(conf);
  }
}

void ASNetwork::forwardingConfiguration(std::map<std::pair<int, int>, fl_t> fl_configurations) {
  for (std::map<std::pair<int, int>, fl_t>::iterator it = fl_configurations.begin(); it != fl_configurations.end(); ++it) {
    ASwitch* as = m_aswitchtable[it->first];     //Pair index
    as->setForwardingLinkDirection(it->second);  //Setting the direction
  }
}

void ASNetwork::childsLinksConfiguration(std::map<std::pair<int, int>, std::pair<bool, bool>> childs_configuration) {
  for (std::map<std::pair<int, int>, std::pair<bool, bool>>::iterator it = childs_configuration.begin(); it != childs_configuration.end(); ++it) {
    ASwitch* as = m_aswitchtable[it->first];
    bool left_child_enabled = std::get<0>(it->second);
    bool right_child_enabled = std::get<1>(it->second);
    as->setChildsEnabled(left_child_enabled, right_child_enabled);
  }
}

void ASNetwork::forwardingToMemoryConfiguration(std::map<std::pair<int, int>, bool> forwarding_to_memory_enabled) {
  for (std::map<std::pair<int, int>, bool>::iterator it = forwarding_to_memory_enabled.begin(); it != forwarding_to_memory_enabled.end(); ++it) {
    ASwitch* as = m_aswitchtable[it->first];
    bool forwarding_to_memory = it->second;
    as->setForwardingToMemoryEnabled(forwarding_to_memory);
  }
}

void ASNetwork::resetSignals() {
  for (std::map<std::pair<int, int>, ASwitch*>::iterator it = m_aswitchtable.begin(); it != m_aswitchtable.end(); ++it) {
    ASwitch* as = it->second;
    as->resetSignals();
  }

  if (this->m_accumulationBufferEnabled > 0) {
    this->p_accumulationBuffer->resetSignals();
  }
}

void ASNetwork::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, std::size_t ms_size, std::size_t n_folding) {
  CompilerART* compiler_art = new CompilerART();  //Creating the object
  compiler_art->configureSignals(current_tile, dnn_layer, ms_size, n_folding);
  std::map<std::pair<int, int>, adderconfig_t> as_signals = compiler_art->get_switches_configuration();
  std::map<std::pair<int, int>, fl_t> as_fw_signals = compiler_art->get_fwlinks_configuration();
  std::map<std::pair<int, int>, std::pair<bool, bool>> as_childs_enabled = compiler_art->get_childs_enabled();
  std::map<std::pair<int, int>, bool> forwarding_to_memory_enabled = compiler_art->get_forwarding_to_memory_enabled();
  this->addersConfiguration(as_signals);
  this->forwardingConfiguration(as_fw_signals);
  this->childsLinksConfiguration(as_childs_enabled);
  this->forwardingToMemoryConfiguration(forwarding_to_memory_enabled);
  if (this->m_accumulationBufferEnabled > 0) {
    this->p_accumulationBuffer->configureSignals(current_tile, dnn_layer, ms_size, n_folding);
  }
  delete compiler_art;
}

void ASNetwork::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, std::size_t ms_size) {
  CompilerART* compiler_art = new CompilerART();  //Creating the object
  compiler_art->configureSparseSignals(sparseVNs, dnn_layer, ms_size);

  std::map<std::pair<int, int>, adderconfig_t> as_signals = compiler_art->get_switches_configuration();
  std::map<std::pair<int, int>, fl_t> as_fw_signals = compiler_art->get_fwlinks_configuration();
  std::map<std::pair<int, int>, std::pair<bool, bool>> as_childs_enabled = compiler_art->get_childs_enabled();
  std::map<std::pair<int, int>, bool> forwarding_to_memory_enabled = compiler_art->get_forwarding_to_memory_enabled();
  this->addersConfiguration(as_signals);
  this->forwardingConfiguration(as_fw_signals);
  this->childsLinksConfiguration(as_childs_enabled);
  this->forwardingToMemoryConfiguration(forwarding_to_memory_enabled);
  delete compiler_art;
}

//TODO Implementar esto bien
void ASNetwork::cycle() {

  //Accumulation buffer cycle if needed
  if (this->m_accumulationBufferEnabled > 0) {
    this->p_accumulationBuffer->cycle();
  }

  //ASNEtwork cycle
  // First are executed the adders that receive the information. This way, the adders uses the data generated in the previous cycle.
  //The order from root to leaves is important in terms of the correctness of the network.
  int switches_this_level = 2;  //Only one switch in the root
  //Going down to the leaves (no count the MSs)
  std::pair<int, int> current_switch_pair(0, 0);  //First node
  ASwitch* as = m_aswitchtable[current_switch_pair];
  as->cycle();
  for (int i = 1; i < this->m_nlevels; i++) {  //From level 1
    int j = 0;
    while (j < switches_this_level) {
      std::pair<int, int> current_switch_pair(i, j);
      ASwitch* as = m_aswitchtable[current_switch_pair];  //Current node j
      //Determining the order of execution if the forwarding link is enabled
      if (as->isFwEnabled()) {                              //If there is fw link in this node
        std::pair<int, int> current_switch_pair(i, j + 1);  //Getting the next which is connected witht his
        ASwitch* as_next = m_aswitchtable[current_switch_pair];
        //Determining the order.
        if (as->getFlDirection() == SEND) {  //If the current one is who send the data, the next goes first since it has to take the data in the next cycle
          as_next->cycle();
          as->cycle();  //executing the cycle for the current DS.
        } else {        //Direction == RECEIVE
          as->cycle();
          as_next->cycle();
        }
        j += 2;  //skip the next one since it has been processed in this iteration

      }       //End fw link is enabled
      else {  //Just this node is executed. The next one and the order between the do not matter at all
        as->cycle();
        j++;  //Not skip since the next one is not performed.
      }
    }
    switches_this_level = switches_this_level * 2;
  }
}

//Print configuration of the ASNetwork
void ASNetwork::printConfiguration(std::ofstream& out, std::size_t indent) {

  out << ind(indent) << "\"ASNetworkConfiguration\" : {" << std::endl;
  //out << ind(indent+IND_SIZE) << "\"ms_size\" : " << this->ms_size  << std::endl; DSNetwork global parameters
  out << ind(indent + IND_SIZE) << "\"ASwitchConfiguration\" : [" << std::endl;  //One entry per DSwitch
  int switches_this_level = 1;
  for (int i = 0; i < this->m_nlevels; i++) {  //From root to leaves (without the MSs)
                                               //Calculating the output ports in this level
    //One array for each level will allow the access to the ASwitch easier
    out << ind(indent + IND_SIZE + IND_SIZE) << "[" << std::endl;
    for (int j = 0; j < switches_this_level; j++) {  // From left to right of the structure
      std::pair<int, int> current_switch_pair(i, j);
      ASwitch* as = m_aswitchtable[current_switch_pair];
      as->printConfiguration(out, indent + IND_SIZE + IND_SIZE + IND_SIZE);
      if (j == (switches_this_level - 1)) {  //If I am in the last switch of the level, the comma to separate the swes is not added
        out << std::endl;                    //This is added because the call to ds print do not show it (to be able to put the comma, if neccesary)
      } else {
        out << "," << std::endl;  //Comma and line break are added to separate with the next ASwitch in the array of this level
      }
    }
    if (i == (this->m_nlevels - 1)) {  //If I am in the last level, the comma to separate the different levels is not added
      out << ind(indent + IND_SIZE + IND_SIZE) << "]" << std::endl;
    }

    else {  //If I am not in the last level, then the comma is printed to separate with the next level
      out << ind(indent + IND_SIZE + IND_SIZE) << "]," << std::endl;
    }

    switches_this_level = switches_this_level * 2;
  }

  out << ind(indent + IND_SIZE) << "]" << std::endl;
  out << ind(indent) << "}";
}

//Printing stats
void ASNetwork::printStats(std::ofstream& out, std::size_t indent) {

  out << ind(indent) << "\"ASNetworkStats\" : {" << std::endl;
  //out << ind(indent+IND_SIZE) << "\"ms_size\" : " << this->ms_size  << std::endl; DSNetwork global parameters
  out << ind(indent + IND_SIZE) << "\"ASwitchStats\" : [" << std::endl;  //One entry per DSwitch
  int switches_this_level = 1;
  for (int i = 0; i < this->m_nlevels; i++) {  //From root to leaves (without the MSs)
                                               //Calculating the output ports in this level
    //One array for each level will allow the access to the ASwitch easier
    out << ind(indent + IND_SIZE + IND_SIZE) << "[" << std::endl;
    for (int j = 0; j < switches_this_level; j++) {  // From left to right of the structure
      std::pair<int, int> current_switch_pair(i, j);
      ASwitch* as = m_aswitchtable[current_switch_pair];
      as->printStats(out, indent + IND_SIZE + IND_SIZE + IND_SIZE);
      if (j == (switches_this_level - 1)) {  //If I am in the last switch of the level, the comma to separate the swes is not added
        out << std::endl;                    //This is added because the call to ds print do not show it (to be able to put the comma, if neccesary)
      } else {
        out << "," << std::endl;  //Comma and line break are added to separate with the next ASwitch in the array of this level
      }
    }
    if (i == (this->m_nlevels - 1)) {  //If I am in the last level, the comma to separate the different levels is not added
      out << ind(indent + IND_SIZE + IND_SIZE) << "]" << std::endl;
    }

    else {  //If I am not in the last level, then the comma is printed to separate with the next level
      out << ind(indent + IND_SIZE + IND_SIZE) << "]," << std::endl;
    }

    switches_this_level = switches_this_level * 2;
  }

  if (this->m_accumulationBufferEnabled > 0) {  //If there is accumulation buffer, then it must be included as a subunit
    out << ind(indent + IND_SIZE) << "]," << std::endl;
  } else {
    out << ind(indent + IND_SIZE) << "]" << std::endl;
  }
  if (this->m_accumulationBufferEnabled > 0) {
    this->p_accumulationBuffer->printStats(out, indent + IND_SIZE);
  }
  out << ind(indent) << "}";
}

void ASNetwork::printEnergy(std::ofstream& out, std::size_t indent) {
  /*
      The ASNetwork component prints the counters for the next subcomponents:
          - ASwitches
          - wires that connect each aswitch with its childs (including the level of wires that connects with the MSNetwork)
          - augmented wires between adders

      Note that the wires that connect with memory are not taken into account in this component. This is done in the CollectionBus.
      Neither are the wires that connect with the accumulation buffer in case this exists. This is done in the AccumulationBuffer.cpp

     */

  //Printing the input wires
  for (std::map<std::pair<int, int>, Connection*>::iterator it = m_inputconnectiontable.begin(); it != m_inputconnectiontable.end(); ++it) {
    Connection* conn = m_inputconnectiontable[it->first];
    conn->printEnergy(out, indent, "RN_WIRE");
  }

  for (std::map<std::pair<int, int>, Connection*>::iterator it = m_forwardingconnectiontable.begin(); it != m_forwardingconnectiontable.end(); ++it) {
    Connection* conn = m_forwardingconnectiontable[it->first];
    conn->printEnergy(out, indent, "RN_WIRE");
  }

  //Printing the ASwitches energy stats and their fifos stats
  for (std::map<std::pair<int, int>, ASwitch*>::iterator it = m_aswitchtable.begin(); it != m_aswitchtable.end(); ++it) {
    ASwitch* as = m_aswitchtable[it->first];  //Pair index
    as->printEnergy(out, indent);             //Setting the direction
  }

  //Printing the accumulator stats
  if (this->m_accumulationBufferEnabled > 0) {
    this->p_accumulationBuffer->printEnergy(out, indent);
  }
}
