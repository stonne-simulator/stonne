//
// Created by Zhongyuan Zhao on 9/14/18.
//
#include "TileGenerator/mRNA/MSNetwork.h"

using namespace mRNA;

MSNetwork::MSNetwork(int pe_size) {
  for(int i = 0; i < pe_size; i++) {
    MSwitch* ms = new MSwitch(i);
    mswitchtable[i] = ms;
  }
  setPhysicalConnection(pe_size);
}

void MSNetwork::setPhysicalConnection(int pe_size) {
  for(int i = 0; i < pe_size; i++) {
    MSwitch* ms = mswitchtable[i];
    if(i == 0) {
      MSwitch* src = mswitchtable[i+1];
      ms->setPhyFInput(src);
    }
    else if(i == pe_size - 1) {
      MSwitch* dst = mswitchtable[i - 1];
      ms->setPhyFOutput(dst);
    }
    else {
      MSwitch* src = mswitchtable[i+1];
      MSwitch* dst = mswitchtable[i-1];
      ms->setPhyFInput(src);
      ms->setPhyFOutput(dst);
    }
  }
}