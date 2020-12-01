#include "CompilerFEN.h"
#include "Tile.h"
#include "utility.h"
#include <math.h>
#include "types.h"
#include <assert.h>
#include "cpptoml.h"

void CompilerFEN::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int num_ms, unsigned int n_folding) {
    assert(ispowerof2(num_ms));
    assert(current_tile->get_VN_Size()*current_tile->get_Num_VNs() <= num_ms);
    this->current_tile = current_tile;
    this->dnn_layer = dnn_layer; //dnn_layer is not used by this compiler
    this->num_ms = num_ms;
    this->n_folding=n_folding;
    this->signals_configured = true;
    //Configuring ART switches
    this->generate_fen_signals(num_ms);
    this->generate_fen_enabling_links(num_ms);
    
}


void CompilerFEN::generate_fen_enabling_links(unsigned int num_ms) {
//Ultimo nivel recorrido aparte hasta el numero de ms
// Los siguientes niveles son recorridos mirando abajo y mirando en la configuracion del nivel inferior. Si tiene hijo 
    std::cout << "Generating ART Enabling links signals" << std::endl;
    int nlevels = log10(num_ms) / log10(2); //All the levels without count the leaves (MSwitches)
    int ms_used = this->current_tile->get_Num_VNs()*this->current_tile->get_VN_Size();
    int num_adders=num_ms / 2; //ALl the adders
    //Last level (not included the ms) apart since we just must look at the ms in order to see if the link is enabled
    for(int i=0; i<num_adders; i++) {
        std::pair<int,int> as_id(nlevels-1, i);
        unsigned int ms_left = i*2;
        unsigned int ms_right = ms_left +1;
        bool left_enabled = false;
        bool right_enabled = false;
        if(ms_left < ms_used) { //if ms_left is within the num_ms used
            left_enabled = true;
        }

        if(ms_right < ms_used) {
            right_enabled = true;
             
        }

        std::pair<bool, bool> as_cfg (left_enabled, right_enabled);
        //Inserting into the map
        childs_enabled[as_id] = as_cfg;
        
    }
    
    //The next levels are evaluated based on the as configuration (ADD_2_1, ADD_3_1,etc.) of the level below
    for(int l=nlevels-2; l>=0; l--) { //Last level was configured previously
        num_adders = num_adders / 2;
        for(int i=0; i<num_adders; i++) {
            std::pair<int,int> as_id(l, i);
            unsigned int as_left = i*2;
            unsigned int as_right = as_left + 1;
            bool left_enabled = false;
            bool right_enabled  = false;
            int previous_level = l+1; //the level below that we look up
            std::pair<int,int> as_left_id (previous_level, as_left);
            std::pair<int,int> as_right_id (previous_level, as_right);
        
            //Evaluating if the left link is enabled
                //Check if the child is in the configuration list. If not, it is not enabled
            if(switches_configuration.count(as_left_id) > 0) {
                 adderconfig_t cfg = switches_configuration[as_left_id]; //Getting the configuration
                 
                 //If the child has one of these configurations it is completely sure that it is sending information to the parent,
                 // which is the node we are evaluating. Therefore, left link must be enabled 
                 if((cfg == ADD_3_1) || (cfg == ADD_1_1_PLUS_FW_1_1) || (cfg == FW_2_2)) {
                     left_enabled = true;
                 }
                 //If the configuration is ADD_2_1, it depends on the fw link. If the fw link is enabled then the child is sending the
                 // the data to that fw link (neighbour node) and therefore the data does not traverse the parent link. 
                 if(cfg == ADD_2_1) {
                     if(fwlinks_configuration.count(as_left_id) == 0) { //If the fw link of the child is NOT enabled
                         left_enabled = true;
                     } //if not, left enabled is false (default)
                     
                 }
            }
                
            //Evaluating if the right link is enabled
               //the very same functionallity as we check with the left link
            if(switches_configuration.count(as_right_id) > 0) {
                 adderconfig_t cfg = switches_configuration[as_right_id]; //Getting the configuration
                 if((cfg == ADD_3_1) || (cfg == ADD_1_1_PLUS_FW_1_1) || (cfg == FW_2_2)) {
                     right_enabled = true;
                 }
                 if(cfg == ADD_2_1) {
                     if(fwlinks_configuration.count(as_right_id) == 0) { //If the fw link of the child is NOT enabled
                         right_enabled = true;
                     } 
                  
                 }
            } //End right evaluation
            //Now we have evaluated both childs, introduce the information to the map
            std::pair<bool, bool> as_cfg (left_enabled, right_enabled);
            childs_enabled[as_id] = as_cfg;
          
            //FORWARDING OPTIMIZATION. TODO What if there is no FW_2_2 because there is a ADDER 2_1 with just one child?
            //Checking if the child has to forward the psum to the collection bus. For doing so, we check if the current node is configured as FW_2_2. This means that the psum have been completed
            //in the previous level since this node just forwards the completed value. 
/*
            adderconfig_t parent_cfg = switches_configuration[as_id];
            if((parent_cfg==FW_2_2) && left_enabled) {
                forwarding_to_memory_enabled[as_left_id] = true;
            }

            if((parent_cfg==FW_2_2) && right_enabled) {
                forwarding_to_memory_enabled[as_right_id] = true;
            } 
  */

            
        } //end of as loop

    } //end of levels loop
}  //end of the function

void CompilerFEN::generate_fen_signals(unsigned int num_ms) {
   std::cout << "Generating FEN signals" << std::endl;
   std::cout << "Num Ms: " << num_ms << std::endl;
   std::cout << "Num VNs: " << this->current_tile->get_Num_VNs() << std::endl;
   std::cout << "VN Size: " << this->current_tile->get_VN_Size() << std::endl;
   int nlevels = log10(num_ms) / log10(2); //All the levels without count the leaves (MSwitches)
   //Vector initialization
   bool* vector_bits = new bool[num_ms];
   bool* parent_bits = new bool[num_ms]; //Vector bits with the parent result. 
   direction_t dir;
   bool fw_to_mem_set=false;
   for(int i=0; i<this->current_tile->get_Num_VNs(); i++) { //For each neuron
        fw_to_mem_set=false;
        //Creating vector for this VN
        for(int j=0; j<num_ms; j++) {
            vector_bits[j]=false;
            parent_bits[j]=false;
        }
        // This VN turned into true
        int shift_this_vn = i*this->current_tile->get_VN_Size(); //Neurons start from 0
        for(int j=shift_this_vn; j<(shift_this_vn+this->current_tile->get_VN_Size()); j++) {
            vector_bits[j]=true; 
        }
	//std::cout << "Vector bits: ";
	//for(int j=0; j<num_ms; j++) {
        //    std::cout << vector_bits[j];
        //}
	//std::cout << std::endl;
        //Iterating over the tree
        int num_adders = num_ms;
        
        for(int l=nlevels-1; l >= 0; l--) { //For each level starting from leaves
            //Iteratinv over the number of adder of that level
            num_adders = num_adders / 2;
            for(int as=0; as < num_adders; as++) {
                std::cout << "Processing ASwitch " << l << ":" << as << std::endl;
                std::cout << "  Vector bits: " << std::endl;
                std::cout << "  ";
                for(int v=0; v<num_ms; v++) {
                    std::cout << vector_bits[v];
                }
                std::cout << std::endl;
                int shift_this_as = as*2; //the first child of this as
                if((as == 0) || (as == (num_adders-1))) {  //The first and the last adders has no fw links 
                    //The options for these nodes are 2:
                    //Option 1: ADD_2_1 if at least one of the childs is 1 and the adder has not been included to the list yet
                    //Option 2: FW_2_2 if at least one of the childs is one and the adder has been included to the list by other neuron (other iter)
                    if(vector_bits[shift_this_as] || vector_bits[shift_this_as+1]) {
                       parent_bits[as] = true; //Updating vector for the next iteration
                       std::pair<int,int> as_id (l,as);
                       if(switches_configuration.count(as_id) > 0) { //Checking if the adder as is in the map
                           switches_configuration[as_id]=FW_2_2;  //FW 2_2 because the adder is already in the list
                       }

                       else {
                           switches_configuration[as_id]=ADD_2_1; //ADD_2_1. If there is only one child, this should work as a fw 1:1
                       }

                    }
                    else { //If both childs are disabled
                        parent_bits[as]=false; //Configuring the vector_bits for the next level
                    }

                }

                else { //If it is a node with fw link
                    //The options of these nodes are 
                    //Option 1: ADD_3_1 FW is enabled and receive and in this direction 
                    if((as % 2)) { //The configuration is done 2 by 2 (pair by pair) when it is the left node of a fw link
                        //Check if the fw link has to be enabled
                        //Left span. Number of 1's from the last child of as to 0
                        bool is_fw_enabled = false; //Indicates if the fw link has to be enabled
                        int left_span = 0;
                        int last_child = shift_this_as + 1; 
			int last_child_enabled = shift_this_as; //Last child with a child enabled
			if(vector_bits[last_child]) {
                            last_child_enabled = last_child;
			}
                        for(int s=last_child_enabled; s>=0; s--) {
                            if(vector_bits[s]) {
                                left_span+=1;
                            }

			    else { //VN must be consecutive nodes
                                break; //Out of the loop
		            }

                        }
                        std::cout << "  Left span: " << left_span << std::endl;
                        //Right span. From the first child of as+1 (i.e., last_child+1) until the end
                        int right_span = 0;
			int first_child_enabled = last_child + 1; // The first node enabled of the next adder. Could be the second. 
			if(!vector_bits[first_child_enabled]) {
                            first_child_enabled+=1; //The second child of the next node
			}
                        for(int s=first_child_enabled; s < num_ms; s++) {  // s= last_child+1
                            if(vector_bits[s]) {
                                right_span+=1;
                            }

			    else { // VN Must be consecutive nodes. If not, the node has no childs and therefore the AD must not be enabled
                                break; //out of the loop
		            }
                        }
                        std::cout << "  Right span: " << right_span << std::endl;                        

                        if((left_span == 0) || (right_span == 0)) { //If it's 0 in one side the fw link is disabled.
                            is_fw_enabled = false;

                            //Updating vector for the next level
			    parent_bits[as]=false; //If both are 0
			    parent_bits[as+1]=false; //If both are 0
                            if(left_span > 0) { //update higher level. It may go to the left or to the parent
                                 
                                parent_bits[as]=true;
                                //Cleaning right bits
                                //for(int s=as+1; s<num_ms; s++) {
                                //    vector_bits[s]=false;
                               // }
                                std::cout << "  Left span is greater than 0 and right span is 0 so aswitch bit " << as << " is set as true and aswitch bit " << as+1 << " as false" << std::endl;
                           //     vector_bits[as+1]=false;
                            }

                            if(right_span > 0) {
                                parent_bits[as+1]=true;
                                //for(int s=0; s<=as; s++) {
                                //    vector_bits[s]=false;
                               // }
                                std::cout << "  Right span is greater than 0 and left span is 0 so aswitch bit " << as+1 << " is set as true and aswitch bit " << as << " as false" << std::endl;
                             //   vector_bits[as]=false;
                            }
                        }
                        
                        else { //Both are not 0 and the algorithm of MAERI paper is applied
                            //Step 1 of the algorithm  (paper) Set direction from smaller to larger span
                            if(left_span >= right_span) { //Equal goes to the left
                                dir = LEFT;
                                std::cout << "  Setting direction to the LEFT" << std::endl;
                            }
                            else {
                                dir = RIGHT;
                                std::cout << "  Setting direction to the RIGHT" << std::endl;
                            }

                            //Step 2. Check if the sub-trees in the direction of the smaller span need to use the parent. If not, enabled.
                            if(dir == LEFT) { //Smaller is to the right
                                if(vector_bits[last_child+3] || vector_bits[last_child+4]) { //we are sure these nodes exist. 
                                    //These are the childs of the next node from the right
				    is_fw_enabled=false; //The parent of the node must be enabled
                                    std::cout << "  Direction is left but, the fw link is NOT enabled as parent is used" << std::endl;
                                    parent_bits[as]=true; //TODO new change
                                    parent_bits[as+1]=true;  //TODO new change
				}
				else {
                                    is_fw_enabled=true;
                                    std::cout << "  Direction is left and the fw link IS ENABLED" << std::endl;
                                    //Cleaning the bits from the nodes right after the right. This is done to update the next level
                                    for(int s=shift_this_as+2; s < num_ms; s++) { //TODO from the first child of the next node
                                        vector_bits[s]=false;
                                    }
                                    parent_bits[as]=true; //There is data here
                                    parent_bits[as+1]=false; // TAKE CARE
                                }
                            }
                            else { //Smaller is to the left since the direction is RIGHT
                                if(vector_bits[last_child-2] || vector_bits[last_child-3]) { //The child of the previous as from the left
				    is_fw_enabled = false; // The parent must be used
                                    std::cout << "  Direction is right but the fw link is NOT enabled as the parent is used" << std::endl;
                                    parent_bits[as]=true; //TODO new change
                                    parent_bits[as+1]=true; //TODO new change
				}
				else {
                                    is_fw_enabled=true;
                                    std::cout << "  Direction is right and the fw link IS ENABLED" << std::endl;
                                    //Cleaning the bits from the left nodes. This is done to update the next level
                                    for(int s=0; s<=shift_this_as+1; s++) {  //TODO From 0 to the last child of the first node 
                                        vector_bits[s]=false;
                                    }
				    parent_bits[as+1]=true;
                                    parent_bits[as]=false; //TAKE CARE
                                }
                            }
                        } //TODO update higher level
                        //Configuration of this AS and the next one using the direction obtained and the fact of wether the fw link has to be enabled
                        if(is_fw_enabled) {
                            //Using as and as[+1] as receiver and sender depending on direction:
                                // as_receive: ADD_3_1
                                // as_send = 
                                //      if(Is not in map): ADD_2_1
                                //      else: ADD_1_1_PLUS_FW_1_1
                             //Calculating who send and who receive
			//	std::cout << "FW IS ENABLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEED" << std::endl;
                             int as_receive; //as that receives the information 
                             int as_send;
                             if(dir == LEFT) {
                                 as_receive = as;
                                 as_send = as+1;
                             }

                             else {
                                 as_receive = as +1;
                                 as_send = as;
                             }
                             std::pair<int,int> as_receive_pair(l, as_receive);
                             std::pair<int,int> as_send_pair(l, as_send);
                             //Enabling FW links
                             fwlinks_configuration[as_receive_pair]=RECEIVE;
                             fwlinks_configuration[as_send_pair]=SEND;
                             switches_configuration[as_receive_pair]=ADD_3_1;
                             std::cout << "  Setting switch " << l << ":" << as_receive << " as ADD_3_1 configuration" << std::endl; 
                             if(switches_configuration.count(as_send_pair) > 0) { //If exist
                                 std::cout << "  Setting switch " << l << ":" << as_send << " as ADD_1_1_PLUS_FW_1_1" << std::endl; 
                                 switches_configuration[as_send_pair]=ADD_1_1_PLUS_FW_1_1;
                             }

                             else {
                                 std::cout << "  Setting switch " << l << ":" << as_send << " as ADD_2_1" << std::endl;
                                 switches_configuration[as_send_pair]=ADD_2_1;
                             }

                        }

                        else { //FW link is not enabled so the options are:
                               // The direction does not matter now.
                               // for t in as and as_next:
                               // if(t is not in map): 
                               //if fw not enabled then  ADD_2_1
                               //else ADD_1_1_PLUS_FW_1_1
                               //else: FW_2_2
                               //Configuring AS
                               std::pair<int,int> current_as_pair (l, as);
                               std::pair<int,int> next_as_pair (l, as+1);
                               std::cout << "  FW link is not enabled so the configuration is done by this other process" << std::endl;
			       if(left_span > 0) { //If there is no left span this AD must be disabled
                                   if(switches_configuration.count(current_as_pair) > 0) {
                                       if(fwlinks_configuration.count(current_as_pair) > 0) { //LAST CHANGE PERFORMED
                                           switches_configuration[current_as_pair]=ADD_1_1_PLUS_FW_1_1; //If fw link is enabled...
                                           assert(fwlinks_configuration[next_as_pair]==SEND);
                                       }
                                       else {
                                           switches_configuration[current_as_pair]=FW_2_2;
                                       }
                                   }

                                   else {
                                       switches_configuration[current_as_pair]=ADD_2_1;
                                   }
                               }
			       if(right_span > 0) { //If there is no right span this AD must be disabled 
                                   //Configuring AS_next
                                   if(switches_configuration.count(next_as_pair) > 0) {
                                       if(fwlinks_configuration.count(next_as_pair) > 0) { //LAST CHANGE PERFORMED
                                           assert(fwlinks_configuration[next_as_pair]==SEND);
                                           switches_configuration[next_as_pair]=ADD_1_1_PLUS_FW_1_1; //If fw link is enabled...
                                       }
                                       else {
                                           switches_configuration[next_as_pair]=FW_2_2;
                                       }
                                   }
                                   else {
                                       switches_configuration[next_as_pair]=ADD_2_1;
                                   }
			       }

                        }

                    } //End if (as % 2)
                } //End else (node is not the first nor the last one)
                
                
            } //End for each node of this level
            //For each level, the bit vector (the one used to work in a certain level) is updated with the parent vector 
            unsigned int n_bits_enabled=0; 
            unsigned int last_bit_enabled;
            for(int s=0; s<num_ms; s++) {
                //First we count the number of switches enabled for the next level (i.e., the number of childs (of the next level) enabled)
                // If there is only one, then that one will be the selected one in charge of sending the output to memory. Since there is just one, we are sure that 
                //this is the switch that performs the last operation of the VN in the ART.
                if(parent_bits[s]==true) { 
                    n_bits_enabled+=1;
                    last_bit_enabled=s;
                }
                vector_bits[s]=parent_bits[s];
                parent_bits[s]=false; //Updating to default for the next iteration
            }

            if(!fw_to_mem_set && (n_bits_enabled==1)) { //If there is just one enabled
                //last_bit_enabled of level l is the switch that is going to send to memory
                std::pair<int,int> switch_to_memory_pair(l, last_bit_enabled); //If there is just one enabled switch, last_bit_enabled contains that only one.
                forwarding_to_fold_node_enabled[switch_to_memory_pair]=true;  //forwarding_to_memory_enabled[switch_to_memory_pair]=true
                fw_to_mem_set=true;
      
                //Configuring the parent that is going to send to memory
                if(l == 0) {
                    std::pair<int,int> as_fw_id(0, 1); //Special node
                    forwarding_to_memory_enabled[as_fw_id]=true; 
                    switches_configuration[as_fw_id]=FOLD;
                }

                else if((last_bit_enabled % 2) == 0) { //If the node is pair, it will use the parent
                    unsigned int fw_as_level = l-1;
                    unsigned int fw_num = last_bit_enabled / 2;
                    std::pair<int,int> as_fw_id(fw_as_level, fw_num);
                    forwarding_to_memory_enabled[as_fw_id]=true; //Enabling the memory bit for the parent
                    switches_configuration[as_fw_id]=FOLD;
                }

                else {
                    unsigned int n_reducciones=0;
                    unsigned int current_node = last_bit_enabled; 
                    while((current_node % 2) != 0) {
                        n_reducciones++;
                        current_node = current_node / 2;
                    }

                   int level_node_to_reduce = l - n_reducciones - 1;
                   if(level_node_to_reduce < 0 ) { // We use the special node to send to memory
                       std::pair<int,int> as_fw_id(0, 1); //Special node
                       forwarding_to_memory_enabled[as_fw_id]=true;                   
                       switches_configuration[as_fw_id]=FOLD;
                   }

                   else {
                       unsigned int node_to_send = current_node / 2;
                       std::pair<int,int> as_fw_id(level_node_to_reduce, node_to_send);
                       forwarding_to_memory_enabled[as_fw_id]=true; //Enabling the memory bit for the parent
                       switches_configuration[as_fw_id]=FOLD;
                   }

              }
              break; //End of the each level loop and we move on to the next neuron
                
                
            }
            
        } //End for each level
    } //End for each VN
    delete[] vector_bits; //Deleting the vector used 
    delete[] parent_bits;

         
}
