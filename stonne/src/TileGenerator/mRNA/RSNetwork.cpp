//
// Created by Zhongyuan Zhao on 9/14/18.
//
#include "TileGenerator/mRNA/RSNetwork.h"

using namespace mRNA;

RSNetwork::RSNetwork(int rn_bandwidth, int pe_size) {
  rn_bw = rn_bandwidth;
  pesize = pe_size;
  int levelnum = log10(pe_size) / log10(2);
  maxlev = levelnum;
  int rs_size = 1;
  for (int l = 0; l < levelnum; l++) {
    for (int i = 0; i < rs_size; i++) {
      RSwitch* rs = new RSwitch(l, i);
      std::pair<int, int> LevandNum(l, i);
      rswitchtable[LevandNum] = rs;
    }
    rs_size = rs_size * 2;
  }
  setPhysicalConnection(levelnum);
}

void RSNetwork::setPhysicalConnection(int levelnum) {
  int fornum = 0;
  int rs_size = 1;
//Set upward link
  for(int l = 0; l < levelnum-1; l++) {
    for(int i = 0; i < rs_size; i++) {
      std::pair<int, int> LevandNum(l, i);
      std::pair<int, int> pair_l(l+1, 2*i);
      std::pair<int, int> pair_r(l+1, 2*i+1);
      std::map<std::pair<int, int>, RSwitch*>::iterator iter_rs = rswitchtable.find(LevandNum);
      std::map<std::pair<int, int>, RSwitch*>::iterator iter_linput = rswitchtable.find(LevandNum);
      std::map<std::pair<int, int>, RSwitch*>::iterator iter_rinput = rswitchtable.find(LevandNum);
      if(iter_rs == rswitchtable.end() || iter_linput == rswitchtable.end() || iter_rinput == rswitchtable.end()) {
        std::cerr << "RSwitch is not created before connection during building the up link of the RSNetwork, "
                  <<"please check the construction function of RSNetwork.\n";
      }
      RSwitch* rs = rswitchtable[LevandNum];
      RSwitch* l_input = rswitchtable[pair_l];
      RSwitch* r_input = rswitchtable[pair_r];
      rs->setPhyInput(l_input, r_input);
      l_input->setPhyOutput(rs);
      r_input->setPhyOutput(rs);
      Forwarder* forwd = new Forwarder(fornum);
      r_input->setPhyFOutput(forwd);
      forwd->setPhyInput(r_input);
      forwardertable[fornum] = forwd;
      fornum++;
    }
    rs_size = rs_size * 2;
  }
//Set augmented link
  rs_size = 1;
  for(int l = 0; l < levelnum; l++) {
    for(int i = 0; i <rs_size; i++) {
      if(i < rs_size -1 && i+1 < rs_size) {
        std::pair<int, int> frontpair(l, i);
        std::pair<int, int> behindpair(l, i+1);
        std::map<std::pair<int, int>, RSwitch*>::iterator iter_front = rswitchtable.find(frontpair);
        std::map<std::pair<int, int>, RSwitch*>::iterator iter_behind = rswitchtable.find(frontpair);
        if(iter_front == rswitchtable.end() || iter_behind == rswitchtable.end()) {
          std::cerr << "RSwitch is not created before connection during building the augmented link of the RSNetwork, "
                    << "please check the construction function of RSNetwork.\n";
        }
        RSwitch* frontrs = rswitchtable[frontpair];
        RSwitch* behindrs = rswitchtable[behindpair];
        RSwitch* frontup = frontrs->getPhyOutput();
        RSwitch* behindup = behindrs->getPhyOutput();
        if(frontup != behindup) {
          frontrs->setPhyAugment(behindrs);
          behindrs->setPhyAugment(frontrs);
        }
      }
    }
    rs_size = rs_size * 2;
  }
}


