//Created 13/06/2019

#include "DSwitch.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <chrono>
#include "utility.h"

using namespace std;

/* This class represents the DSwitch of the MAERI architecture. Basically, the class contains to connections, which   */

DSwitch::DSwitch(id_t id, std::string name, unsigned int level, unsigned int num_in_level, Config stonne_cfg, unsigned int ms_size) : Unit(id, name)  { //ms_size depends on the local ms_size in the subtree
    this->level = level;
    this->num_in_level = num_in_level;
    this->num_ms = ms_size;
    this->time_routing = 0;
    this->pending_data = false;
    this->current_capacity=0;

    this->latency=stonne_cfg.m_DSwitchCfg.latency;
    this->input_ports=stonne_cfg.m_DSwitchCfg.input_ports;
    this->output_ports=stonne_cfg.m_DSwitchCfg.output_ports;
    this->port_width=stonne_cfg.m_DSwitchCfg.port_width;

    
}

DSwitch::DSwitch(id_t id, std::string name, unsigned int level, unsigned int num_in_level, Config stonne_cfg, unsigned int ms_size, 
    Connection* leftConnection, Connection* rightConnection, Connection* inputConnection) :DSwitch(id, name, level, num_in_level, stonne_cfg, ms_size) {
    this->current_capacity = 0; 
    this->pending_data=false;
    this->setLeftConnection(leftConnection);
    this->setRightConnection(rightConnection);
    this->setInputConnection(inputConnection);
}

void DSwitch::setLeftConnection(Connection *leftConnection) {
    this->leftConnection = leftConnection;
}

void DSwitch::setRightConnection(Connection *rightConnection) {
    this->rightConnection = rightConnection;
}

void DSwitch::setInputConnection(Connection *inputConnection) {
    this->inputConnection = inputConnection;
}


//Send a package to left or right connection. If there is no remaining bandiwth in the connection an exception is raised
void DSwitch::send(vector<DataPackage*> data_p, Connection* connection) {
    assert(this->pending_data); //There must exist data in the switch
    connection->send(data_p); //Send the data to the corresponding output
}

//Differently from connection, the receive in the switch means get the data from the connection and save it. 
void DSwitch::receive() { 
    //TODO Check the bw!
    if(this->inputConnection->existPendingData()) { //If there is data to receive
    	this->data = this->inputConnection->receive(); //Copying the data to receive
        this->pending_data = true;
    }
    return;
}

/*
 Auxiliary function used to route and send the packages. The function does 3 distinctions:
    1. Broadcast: The package is replicated and sent through the output left and output right links
    2. Unicast: The package is sent either to the right or to the left depending on unicast_dest value, which is the destination
    3. Multicast: Using the array dests, all the destinations are evaluated. If there is more than a package in one direction, the package must be sent
*/
void DSwitch::route_packages() { //TODO It is supposing you have enouth bandwidth in the connection
    //Vector used to send the packages
    vector<DataPackage*> output_left;    // List of packages to send to the left output connection
    vector<DataPackage*> output_right;   // List of packages to send to the right output connection
  
    //Iteration over the packages
    for(int i=0; i<this->data.size(); i++) { //For every package in the connection
        DataPackage* pck = this->data[i]; //To repeat the package

        //If the package is broadcast, the package is replicated and push in to both links
        if(pck->isBroadcast()) {  //If the package is broacast, no routing is needed.
           // The package is send to both by replicating the pointers.
           DataPackage* pck_left = new DataPackage(pck);
           DataPackage* pck_right = new DataPackage(pck);
           output_left.push_back(pck_left);
           output_right.push_back(pck_right);
        }

         else if(pck->isUnicast()) {
             unsigned int n_ms_to_route = num_ms >> (this->level); // num_ms  that switch can route  (num_ms / 2^level). level starts in 0
             unsigned int first_ms_to_route = this->num_in_level*n_ms_to_route; //Shift of that certain switch. num_in_level starts in 0. 
             //unsigned int last_ms_to_route = first_ms_to_route + n_ms_ro_route - 1; //The index in the array is -1 
             float half_ms_to_route = (first_ms_to_route + (n_ms_to_route-1)/2.0); // -1 to get the intermedium value between left and right
             //Now we can route. if the destination is smaller than half then left. if it is greater right. 
             // Note half is decimal value. This is so to distinguish between both branches.
             if(pck->get_unicast_dest() < half_ms_to_route) {
                 DataPackage* pck_left = new DataPackage(pck);
                 output_left.push_back(pck_left);
             }

             else { //(pck->get_unicast_dest() > half_ms_to_route) 
                 DataPackage* pck_right = new DataPackage(pck);
                 output_right.push_back(pck_right);
             }
              
         }

         else if(pck->isMulticast()) {
         //    auto start = std::chrono::steady_clock::now();
             unsigned int n_ms_to_route = num_ms >> (this->level); // num_ms  that switch can route  (num_ms / 2^level). level starts in 0
             unsigned int first_ms_to_route = this->num_in_level*n_ms_to_route; //Shift of that certain switch. num_in_level starts in 0. 
             unsigned int last_ms_to_route = first_ms_to_route + n_ms_to_route; //The index in the array is -1 
          
             
             float half_ms_to_route = (first_ms_to_route + (n_ms_to_route-1)/2.0); //- 1 to get the intermedium value between left and right
             bool route_right = false;
             bool route_left = false;

             //Left half exploration
             int half_lim = (int) floor(half_ms_to_route); //The last index of the left half i.e., half_ms_to_route -0.5
             const bool* dests = pck->get_dests(); //Be aware since the array belongs to pck. Must not be delete!
             for(int i=first_ms_to_route; i <=  half_lim; i++) {
                 if(dests[i]) { //If ms i needs to receive the message
                     route_left = true;
                     break; //Out the for. Once one of element of that side has to be send to that branch, the rest do not matter.
                 }
             }
             
             // Right half exploration
             
             for(int i=half_lim+1; i < last_ms_to_route; i++) { //The same for the right part. Notice the index are correct
                 if(dests[i]) {
                     route_right = true; //Right enabled
                     break;
                 }
             }
             //auto end = std::chrono::steady_clock::now();
             //this->time_routing+=std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

             if(route_left) { //Pushing the package into the output left link
                 DataPackage* pck_left = new DataPackage(pck);
                 output_left.push_back(pck_left);
             }

             if(route_right) { //Pushing the package into the output right link
                // auto start = std::chrono::steady_clock::now();
                 DataPackage* pck_right = new DataPackage(pck);
                 output_right.push_back(pck_right);
                //  auto  end = std::chrono::steady_clock::now();
                // this->time_routing+=std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
             }
           //  auto  end = std::chrono::steady_clock::now();
            // this->time_routing+=std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

         }

         else {
             assert(false); //This case is not possible
         }//auto start = std::chrono::steady_clock::now();
         delete pck;

    }

    if((output_left.size() > 0) && (output_right.size() > 0)) { //Broadcast
       this->dswitchStats.n_broadcasts++; //Increase the counter 
    }

    else if((output_left.size() > 0) ||  (output_right.size() > 0)) {
        this->dswitchStats.n_unicasts++;
    }
   

    //Sending data to the branches
    if(output_left.size() > 0) { //If in the left branch there is something to send
        this->send(output_left, this->leftConnection);
        this->dswitchStats.n_left_sends++;
    }
    if(output_right.size() > 0) { //If in the right branch there is something to send
        this->send(output_right, this->rightConnection);
        this->dswitchStats.n_right_sends++;
    }


    this->pending_data = false; //Supposing you have enought bandwidth, which is always true in a chubby tree
    
}


void DSwitch::cycle() {
    this->dswitchStats.total_cycles++; //To track information
    //Send depending on routing each package to its corrresponding output left or right
    //Steps to follow:
    // 1. Get data from input connection using the function receive()
    // 2. If pending_data, read package by package and get the destination using the header
    // 3. Group inputs depending on the destination into two output groups
    // 4. Send the vectors to its corresponding outptus. 
    this->receive();
    //Iterate over every package
    if(this->pending_data) {//If there is data to send 
        this->route_packages();
    }
    

}

void DSwitch::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    this->dswitchStats.print(out, indent+IND_SIZE);
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void DSwitch::printEnergy(std::ofstream& out, unsigned int indent) {

    /*
         This component prints:
             - The number of unicasts routed
             - The number of broadcasts routed
    */
    out << ind(indent) << "SWITCH ROUTE_UNICAST=" << this->dswitchStats.n_unicasts;
    out << ind(indent) << " ROUTE_BROADCAST=" << this->dswitchStats.n_broadcasts << std::endl;
}


