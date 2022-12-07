//
// Created by Zhongyuan Zhao on 9/12/18.
//
#include "TileGenerator/mRNA/MAERIModel.h"
using namespace mRNA;

void Maeri::ConnectMSNandDSN() {
  int level = dsnet->getmaxlev();
  int pesize = dsnet->getpesize();
  for (int i =0; i < pesize; i++) {
    std::pair<int, int> levandnum(level-1, i);
    DSwitch* ds = dsnet->dswitchtable[levandnum];
    if(i % 2 == 0) {
      MSwitch* lms = msnet->mswitchtable[i];
      MSwitch* rms = msnet->mswitchtable[i+1];
      ds->setPhyOutput(lms, rms);
      std::pair<int, int> rdspair(level-1, i+1);
      DSwitch* rds = dsnet->dswitchtable[rdspair];
      lms->setPhyInput(ds, rds);
      rms->setPhyInput(ds, rds);
    }
    else {
      MSwitch* lms = msnet->mswitchtable[i-1];
      MSwitch* rms = msnet->mswitchtable[i];
      ds->setPhyOutput(lms, rms);
    }
  }
}

void Maeri::ConnectMSNandRSN() {
  int fwdnum = rsnet->forwardertable.size();
  int level = rsnet->getmaxlev();
  int pesize = rsnet->getpesize();
  for(int i = 0; i < pesize; i = i+2) {
    int rsnum = i / 2;
    MSwitch* lms = msnet->mswitchtable[i];
    MSwitch* rms = msnet->mswitchtable[i+1];
    std::pair<int, int> rspair(level -1, rsnum);
    RSwitch* rs = rsnet->rswitchtable[rspair];
    lms->setPhyOutput(rs);
    rms->setPhyOutput(rs);
    rs->setPhyInput(lms, rms);
    Forwarder* forwd = new Forwarder(fwdnum);
    rms->setPhyFOutput(forwd);
    forwd->setPhyInput(rms);
    rsnet->forwardertable[fwdnum] = forwd;
    fwdnum++;
  }
}

void Maeri::DrawNetwork(std::ofstream& profile) {
  int maxlevel = dsnet->getmaxlev();
  std::vector<MSwitch*> recordms;
  profile << "digraph G " << "{\n{\n";
  profile << "\tlabel=\"" << "DSNetwork" << "\";\n";

  for(std::map<std::pair<int, int>, DSwitch*>::iterator I = dsnet->dswitchtable.begin(), E = dsnet->dswitchtable.end(); I != E; I++) {
    std::pair<int, int> levandnum = I->first;
    DSwitch* ds = I->second;
    int level = levandnum.first;
    int number  = levandnum.second;
    profile << "\tNode" << ds << "[shape=circle, style=filled, width=.3, color=green,";
    profile << "label=\"{" << "DS L" << level << "_N" << number << "}\"];\n";
    DSwitch* ldst = ds->getPhyLOutput();
    DSwitch* rdst = ds->getPhyROutput();
    if(ldst){
      profile << "\tNode" << ds << " -> " << "Node" << ldst << ";\n";
    }
    else {
      MSwitch* lms = ds->getPhyLMS();
      if(lms == NULL) {
        std::cerr << "A DSwitch must either connect to a DSwitch or MSwitch. Please check the physical connection function.\n";
      }
      std::vector<MSwitch*>::iterator iter = std::find(recordms.begin(), recordms.end(), lms);
      if(iter == recordms.end()) {
        int msnum = lms->getmsnum();
        profile << "\tNode" << lms << "[shape=box, style=filled, color=red,";
        profile << "label=\"{" << "MS " << msnum << "}\"];\n";
        profile << "\tNode" << ds << " -> " << "Node" << lms << ";\n";
      }
    }

    if(rdst){
      profile << "\tNode" << ds << " -> " << "Node" << rdst << ";\n";
    }
    else {
      MSwitch* rms = ds->getPhyRMS();
      if(rms == NULL) {
        std::cerr << "A DSwitch must either connect to a DSwitch or MSwitch. Please check the physical connection function.\n";
      }
      std::vector<MSwitch*>::iterator iter = std::find(recordms.begin(), recordms.end(), rms);
      if(iter == recordms.end()) {
        int msnum = rms->getmsnum();
        profile << "\tNode" << rms << "[shape=record, style=filled, color=red,";
        profile << "label=\"{" << "MS_" << msnum << "}\"];\n";
        profile << "\tNode" << ds << " -> " << "Node" << rms << ";\n";
      }
    }
  }

/*  profile << "\t{ rank=same; ";
  for(int i = 0; i < pe_size; i++) {
    MSwitch* ms = msnet->mswitchtable[i];
    profile << "\"" << "Node" << ms << "\";";
  }
  profile << "}\n";
*/

  for(int i = 0; i < pe_size; i++) {
    MSwitch* ms = msnet->mswitchtable[i];
    RSwitch* rs = ms->getPhyOutput();
    int rslev = rs->getrslev();
    int rsnum = rs->getrsnum();
    if(i % 2 == 0) {
      profile << "\tNode" << rs << "[shape=box, style=filled, color=lightblue,";
      profile << "label=\"{" << "RS L" << rslev << "_N" << rsnum << "}\"];\n";
    }
    else {
      Forwarder* fwd = ms->getPhyFOutput();
      if(fwd == NULL) {
        std::cerr << "MSwitch who has odd number should be connected to forwarder, please check the connectmsandrs function.";
      }
      int num = fwd->getfdnum();
      profile << "\tNode" << fwd << "[shape=circle, style=filled, color=orange,";
      profile << "label=\"{" << "FD " << num << "}\"];\n";
      profile << "\tNode" << ms << " -> " << "Node" << fwd << ";\n";
    }
    profile << "\tNode" << ms << " -> " << "Node" << rs << ";\n";

    MSwitch* leftms = ms->getPhyMInput();
    if(leftms) {
      //profile << "\tedge [color=red];\n";
      profile << "\tNode" << leftms << " -> " << "Node" << ms << ";\n";
    }
  }

  for(std::map<std::pair<int, int>, RSwitch* >::iterator I = rsnet->rswitchtable.begin(), E = rsnet->rswitchtable.end(); I != E; I++) {
    std::pair<int, int> levandnum = I->first;
    RSwitch* rs = I->second;
    int lev = levandnum.first;
    int num = levandnum.second;
    if(lev != maxlevel - 1) {
      profile << "\tNode" << rs << "[shape=box, style=filled, color=lightblue,";
      profile << "label=\"{" << "RS L" << lev << "_N" << num << "}\"];\n";
    }

    if(lev != 0) {
      RSwitch* up_rs = rs->getPhyOutput();
      profile << "\tNode" << rs << " -> " << "Node" << up_rs << ";\n";
      if(rs->getPhyAOutput() != NULL) {
        RSwitch* augmtrs = rs->getPhyAOutput();
        profile << "\tNode" << rs << " -> " << "Node" << augmtrs << ";\n";
      }
      if(rs->getPhyFOutput() != NULL) {
        Forwarder* fw = rs->getPhyFOutput();
        int fwdnum = fw->getfdnum();
        profile << "\tNode" << fw << "[shape=circle, style=filled, color=orange,";
        profile << "label=\"{" << "FD " << fwdnum << "}\"];\n";
        profile << "\tNode" << rs << " -> " << "Node" << fw << ";\n";
      }
    }
  }

  /*profile << "\t{ rank=same; ";
  for(int i = 0; i < pe_size; i++) {
    MSwitch* ms = msnet->mswitchtable[i];
    profile << "\"" << "Node" << ms << "\";";
  }
  profile << "};\n";
  */


  profile << "}\n}\n";
}
