//
// Created by Zhongyuan Zhao on 9/14/18.
//
#include "TileGenerator/mRNA/DSNetwork.h"

using namespace mRNA;

void DSwitch::ConfigDS(int input_src, int output_mode) {
  if(input_src == 0) {
    config_input = input_lds;
  }
  else {
    config_rds = input_rds;
  }

  if(output_mode == 0) {
    config_lds = output_lds;
  }
  else if(output_mode == 1) {
    config_rds = output_rds;
  }
  else {
    config_lds = output_lds;
    config_rds = output_rds;
  }
}

int DSwitch::getAvOutput() {
  int i = 0;
  return i;
}

DSNetwork::DSNetwork(double bw, int pe_size) {
  bandwidth = bw;
  int levelnum = log10(pe_size)/log10(2);
  maxlev = levelnum;
  for(int i = 0; i < levelnum; i++) {
    for(int j = 0; j < pe_size; j++) {
      DSwitch* ds = new DSwitch(i, j);
      std::pair<int, int> LevelandNum(i, j);
      dswitchtable[LevelandNum] = ds;
    }
  }
  setPhysicalConnection(levelnum, pe_size);
}

//TODO: This part is a heuristic exploration, we can set different connectivity
//to see the topology influence on DSwitch

void DSNetwork::setPhysicalConnection(int levelnum, int pe_size) {
  rootSWsize = pe_size;
  int segnum = 1;
  for(int i = 0; i < levelnum-1; i++) {
    segnum = segnum * 2;
    int seg = pe_size / segnum;
    for(int t = 0; t < segnum; t++) {
      for(int j = 0; j < seg; j++) {
        int num = t * seg + j;
        std::pair<int, int> srcds(i, num);
        DSwitch* src = dswitchtable[srcds];
        if(t % 2 == 0) {
          int leftnum = num;
          int rightnum = num + seg;
          std::pair<int, int> dstl(i+1, leftnum);
          std::pair<int, int> dstr(i+1, rightnum);
          DSwitch* dstdsl = dswitchtable[dstl];
          DSwitch* dstdsr = dswitchtable[dstr];
          src->setPhyOutput(dstdsl, dstdsr);
          dstdsl->setPhyLInput(src);
          dstdsr->setPhyLInput(src);
        }
        else {
          int rightnum = num;
          int leftnum = 2 * t * seg - 1 - num;
          std::pair<int, int> dstl(i+1, leftnum);
          std::pair<int, int> dstr(i+1, rightnum);
          DSwitch* dstdsl = dswitchtable[dstl];
          DSwitch* dstdsr = dswitchtable[dstr];
          src->setPhyOutput(dstdsl, dstdsr);
          dstdsl->setPhyRInput(src);
          dstdsr->setPhyRInput(src);
        }
      }
    }
  }
}

void DSNetwork::rootSWsread(int* array, int length, int interval) {
  int lev = 0;
  int num = 0;
  int vnid = 0;
  for (int i = 0; i < length; i++) {
    std::pair<int, int> levandnum(lev, num);
    DSwitch* ds = dswitchtable[levandnum];
    ds->readdata(array[i]);
    ds->setvnid(vnid);
    num = num + interval;
    if(num > rootSWsize) {
      std::cerr << "The number of needed switches exceeds the total number of root switches, reset the array size or interval value.\n";
    }
  }
}
