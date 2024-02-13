#ifndef MAERIMODEL_H_
#define MAERIMODEL_H_

#include <algorithm>
#include <string>
#include "DSNetwork.hpp"
#include "MSNetwork.hpp"
#include "RSNetwork.hpp"
#include "SDMemory.hpp"

namespace mRNA {

class Maeri {
 public:
  Maeri(int pesize, int dn_bandwidth, int rn_bandwidth) {
    pe_size = pesize;
    dn_bw = dn_bandwidth;
    rn_bw = rn_bandwidth;
    msnet = new MSNetwork(pesize);
    dsnet = new DSNetwork(dn_bandwidth, pesize);
    rsnet = new RSNetwork(rn_bandwidth, pesize);
    ConnectMSNandDSN();
    ConnectMSNandRSN();
  }

  MSNetwork* msnet;
  DSNetwork* dsnet;
  RSNetwork* rsnet;
  SDMemory* sdm;
  int pe_size;
  int dn_bw;
  int rn_bw;
  void ConnectMSNandDSN();
  void ConnectMSNandRSN();
  //This function used to test the physical connection of the distribute network;
  void DrawNetwork(std::ofstream& profile);
};
}  // namespace mRNA

#endif
//TO DO add enumerate.
