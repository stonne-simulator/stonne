//
// Created by Zhongyuan Zhao on 9/27/18.
//

#include "TileGenerator/mRNA/MappingStrategy.h"

using namespace mRNA;

void MappingStrategy::setvn_num(int i, int vnnum) {
  if(vn_num.find(i) == vn_num.end()) {
    vn_num[i] = vnnum;
  }
  else {
    std::cerr << "The value vn_num in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setelementinvn(int i, int element) {
  if(elementinvn.find(i) == elementinvn.end()) {
    elementinvn[i] = element;
  }
  else {
    std::cerr << "The value elementinvn in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setdn_access(int i, unsigned long long dn) {
  if(dn_access.find(i) == dn_access.end()) {
    dn_access[i] = dn;
  }
  else {
    std::cerr << "The value dn_access in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setinput_unicast(int i, int inputunicast) {
  if (input_unicast.find(i) == input_unicast.end() || i == 0) {
    input_unicast[i].push_back(inputunicast);
  }
  else {
    std::cerr << "The value input_unicast in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setweight_unicast(int i, int weightunicast) {
  if(weight_unicast.find(i) == weight_unicast.end() || i == 0) {
    weight_unicast[i].push_back(weightunicast);
  }
  else {
    std::cerr << "The value weight_unicast in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setinput_multicast(int i, int inputmulticast) {
  if(input_multicast.find(i) == input_multicast.end() || i == 0) {
    input_multicast[i].push_back(inputmulticast);
  }
  else {
    std::cerr << "The value input_multicast in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setweight_multicast(int i, int weightmulticast) {
  if(weight_multicast.find(i) == weight_multicast.end() || i == 0) {
    weight_multicast[i].push_back(weightmulticast);
  }
  else {
    std::cerr << "The value weight_multicast in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setweightmulticast_size(int i, int size) {
  if(weightmulticast_size.find(i) == weightmulticast_size.end() || i == 0) {
    weightmulticast_size[i].push_back(size);
  }
  else {
    std::cerr << "The size of weight multicast in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setinputmulticast_size(int i, int size) {
  if(inputmulticast_size.find(i) == inputmulticast_size.end() || i == 0) {
    inputmulticast_size[i].push_back(size);
  }
  else {
    std::cerr << "The size of weight_multicast in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setmslink_access(int i, unsigned long long mslink) {
  if(mslink_access.find(i) == mslink_access.end()) {
    mslink_access[i] = mslink;
  }
  else {
    std::cerr << "The value mslink_access in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setmsreg_access(int i, unsigned long long msreg) {
  if(msreg_access.find(i) == msreg_access.end()) {
    msreg_access[i] = msreg;
  }
  else {
    std::cerr << "The value msreg_access in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setmultiply_num(int i, unsigned long long multiply) {
  if(multiply_num.find(i) == multiply_num.end()) {
    multiply_num[i] = multiply;
  }
  else {
    std::cerr << "The value multiply_num in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setrn_access(int i, unsigned long long rn) {
  if(rn_access.find(i) == rn_access.end()) {
    rn_access[i] = rn;
  }
  else {
    std::cerr << "The value rn_access in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setreduce_num(int i, unsigned long long reduce) {
  if(reduce_num.find(i) == reduce_num.end()) {
    reduce_num[i] = reduce;
  }
  else {
    std::cerr << "The value reduce_num in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setspminput_read(int i, unsigned long long read) {
  if(spminput_read.find(i) == spminput_read.end()) {
    spminput_read[i] = read;
  }
  else {
    std::cerr << "The value spminput_read in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setspmweight_read(int i, unsigned long long read) {
  if(spmweight_read.find(i) == spmweight_read.end()) {
    spmweight_read[i] = read;
  }
  else {
    std::cerr << "The value spmweight_read in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setspm_write(int i, unsigned long long write) {
  if(spm_write.find(i) == spm_write.end()) {
    spm_write[i] = write;
  }
  else {
    std::cerr << "The value spm_write in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setcode_size(int i, int size) {
  if(code_size.find(i) == code_size.end()) {
    code_size[i] = size;
  }
  else {
    std::cerr << "The value code_size in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setcontrol_step(int i, unsigned long long cs) {
  if(config_cs.find(i) == config_cs.end()) {
    config_cs[i] = cs;
  }
  else {
    std::cerr << "The value config_cs in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setcycle(int i, unsigned long long cycle) {
  if(config_cycle.find(i) == config_cycle.end()) {
    config_cycle[i] = cycle;
  }
  else {
    std::cerr << "The value config_cycle in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setinput_trans(int i, unsigned long long trans) {
  if(input_trans.find(i) == input_trans.end()) {
    input_trans[i] = trans;
  }
  else {
    std::cerr << "The value input_trans in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setfilter_trans(int i, unsigned long long trans) {
  if(filter_trans.find(i) == filter_trans.end()) {
    filter_trans[i] = trans;
  }
  else {
    std::cerr << "The value filter_trans in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setpeak_ur(int i, double ur) {
  if(peak_ur.find(i) == peak_ur.end()) {
    peak_ur[i] = ur;
  }
  else {
    std::cerr << "The value peak_ur in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setconfig_ur(int i, double ur) {
  if(config_ur.find(i) == config_ur.end()) {
    config_ur[i] = ur;
  }
  else {
    std::cerr << "The value config_ur in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setconfig_attribute(int i, ConfigType attribute) {
  if(config_attribute.find(i) == config_attribute.end()) {
    config_attribute[i] = attribute;
  }
  else {
    std::cerr << "The attribute in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setspminputread_energy(int i, double spminputread) {
  if(spminputread_eng.find(i) == spminputread_eng.end()) {
    spminputread_eng[i] = spminputread;
  }
  else {
    std::cerr << "The spminputread_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::setspmweightread_energy(int i, double spmweightread) {
  if(spmweightread_eng.find(i) == spmweightread_eng.end()) {
    spmweightread_eng[i] = spmweightread;
  }
  else {
    std::cerr << "The spmweightread_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}
void MappingStrategy::setspmwrite_energy(int i, double spmwrite) {
  if(spmwrite_eng.find(i) == spmwrite_eng.end()) {
    spmwrite_eng[i] = spmwrite;
  }
  else {
    std::cerr << "The spmwrite_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}
void MappingStrategy::setmsreg_energy(int i, double msreg) {
  if(msreg_eng.find(i) == msreg_eng.end()) {
    msreg_eng[i] = msreg;
  }
  else {
    std::cerr << "The msreg_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}
void MappingStrategy::setmslink_energy(int i, double mslink) {
  if(msaccess_eng.find(i) == msaccess_eng.end()) {
    msaccess_eng[i] = mslink;
  }
  else {
    std::cerr << "The msaccess_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}
void MappingStrategy::setmultiply_energy(int i, double multiply) {
  if(multiply_eng.find(i) == multiply_eng.end()) {
    multiply_eng[i] = multiply;
  }
  else {
    std::cerr << "The multiply_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}
void MappingStrategy::setrsaccess_energy(int i, double rsaccess) {
  if(rsaccess_eng.find(i) == rsaccess_eng.end()) {
    rsaccess_eng[i] = rsaccess;
  }
  else {
    std::cerr << "The rsaccess_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}
void MappingStrategy::setreduce_energy(int i, double reduce) {
  if(reduce_eng.find(i) == reduce_eng.end()) {
    reduce_eng[i] = reduce;
  }
  else {
    std::cerr << "The reduce_eng in configuration " << i << " has been set before, please check analyzer.\n";
  }
}
void MappingStrategy::setdnaccess_energy(int i, double dnaccess) {
  if(dsaccess_eng.find(i) == dsaccess_eng.end()) {
    dsaccess_eng[i] = dnaccess;
  }
  else {
    std::cerr << "The attribute in configuration " << i << " has been set before, please check analyzer.\n";
  }
}

void MappingStrategy::sum_all() {

  int totalcodesize = 0;
  unsigned long long totalcs = 0;
  unsigned long long totalcycle = 0;
  unsigned long long totalinput_trans = 0;
  unsigned long long totalfilter_trans = 0;
  unsigned long long totaldn = 0;
  unsigned long long totalmslink = 0;
  unsigned long long totalmsreg = 0;
  unsigned long long totalmultiply = 0;
  unsigned long long totalreduce = 0;
  unsigned long long totalrn = 0;
  unsigned long long totalspminputread = 0;
  unsigned long long totalspmweightread = 0;
  unsigned long long totalspmwrite = 0;

  double totaldneng = 0;
  double totalmslinkeng = 0;
  double totalmsregeng = 0;
  double totalmultiplyeng = 0;
  double totalreduceeng = 0;
  double totalrneng = 0;
  double totalspminputreadeng = 0;
  double totalspmweightreadeng = 0;
  double totalspmwriteeng = 0;

  for(int i = 0; i < config_num; i++) {
    totalcodesize += code_size[i];
    totalcs += config_cs[i];
    totalcycle += config_cycle[i];
    totalinput_trans += input_trans[i];
    totalfilter_trans += filter_trans[i];
    totaldn += dn_access[i];
    totalmslink += mslink_access[i];
    totalmsreg += msreg_access[i];
    totalmultiply += multiply_num[i];
    totalreduce += reduce_num[i];
    totalrn += rn_access[i];
    totalspminputread += spminput_read[i];
    totalspmweightread +=spmweight_read[i];
    totalspmwrite += spm_write[i];
    totaldneng += dsaccess_eng[i];
    totalmslinkeng += msaccess_eng[i];
    totalmsregeng += msreg_eng[i];
    totalmultiplyeng += multiply_eng[i];
    totalreduceeng += reduce_eng[i];
    totalrneng += rsaccess_eng[i];
    totalspminputreadeng += spminputread_eng[i];
    totalspmweightreadeng +=spmweightread_eng[i];
    totalspmwriteeng += spmwrite_eng[i];

  }

  if(code_size.find(config_num) == code_size.end()) {
    code_size[config_num] = totalcodesize;
  }
  else {
    std::cerr << "The total code size should not be set before, please check analyzer.\n";
  }

  if(config_cs.find(config_num) == config_cs.end()) {
    config_cs[config_num] = totalcs;
  }
  else {
    std::cerr << "The total control step should not be set before, please check analyzer.\n";
  }

  if(config_cycle.find(config_num) == config_cycle.end()) {
    config_cycle[config_num] = totalcycle;
  }
  else {
    std::cerr << "The total cycle should not be set before, please check analyzer.\n";
  }

  if(input_trans.find(config_num) == input_trans.end()) {
    input_trans[config_num] = totalinput_trans;
  }
  else {
    std::cerr << "The total input trans should not be set before, please check analyzer.\n";
  }

  if(filter_trans.find(config_num) == filter_trans.end()) {
    filter_trans[config_num] = totalfilter_trans;
  }
  else {
    std::cerr << "The total filter trans should not be set before, please check analyzer.\n";
  }

  if(dn_access.find(config_num) == dn_access.end()) {
    dn_access[config_num] = totaldn;
  }
  else {
    std::cerr << "The total dn access should not be set before, please check analyzer.\n";
  }

  if(mslink_access.find(config_num) == mslink_access.end()) {
    mslink_access[config_num] = totalmslink;
  }
  else {
    std::cerr << "The total mslink access should not be set before, please check analyzer.\n";
  }

  if(msreg_access.find(config_num) == msreg_access.end()) {
    msreg_access[config_num] = totalmsreg;
  }
  else {
    std::cerr << "The total msreg access should not be set before, please check analyzer.\n";
  }

  if(multiply_num.find(config_num) == multiply_num.end()) {
    multiply_num[config_num] = totalmultiply;
  }
  else {
    std::cerr << "The total multiply number should not be set before, please check analyzer.\n";
  }

  if(reduce_num.find(config_num) == reduce_num.end()) {
    reduce_num[config_num] = totalreduce;
  }
  else {
    std::cerr << "The total reduce operation should not be set before, please check analyzer.\n";
  }

  if(rn_access.find(config_num) == rn_access.end()) {
    rn_access[config_num] = totalrn;
  }
  else {
    std::cerr << "The total rn access should not be set before, please check analyzer.\n";
  }

  if(spminput_read.find(config_num) == spminput_read.end()) {
    spminput_read[config_num] = totalspminputread;
  }
  else {
    std::cerr << "The total spm input read should not be set before, please check analyzer.\n";
  }

  if(spmweight_read.find(config_num) == spmweight_read.end()) {
    spmweight_read[config_num] = totalspmweightread;
  }
  else {
    std::cerr << "The total spm weight read should not be set before, please check analyzer.\n";
  }

  if(spm_write.find(config_num) == spm_write.end()) {
    spm_write[config_num] = totalspmwrite;
  }
  else {
    std::cerr << "The total spm write should not be set before, please check analyzer.\n";
  }

  if(spminputread_eng.find(config_num) == spminputread_eng.end()) {
    spminputread_eng[config_num] = totalspminputreadeng;
  }
  else {
    std::cerr << "The total spm input read energy should not be set before, please check analyzer.\n";
  }

  if(spmweightread_eng.find(config_num) == spmweightread_eng.end()) {
    spmweightread_eng[config_num] = totalspmweightreadeng;
  }
  else {
    std::cerr << "The total spm weight read energy should not be set before, please check analyzer.\n";
  }

  if(spmwrite_eng.find(config_num) == spmwrite_eng.end()) {
    spmwrite_eng[config_num] = totalspmwriteeng;
  }
  else {
    std::cerr << "The total spm write energy should not be set before, please check analyzer.\n";
  }

  if(rsaccess_eng.find(config_num) == rsaccess_eng.end()) {
    rsaccess_eng[config_num] = totalrneng;
  }
  else {
    std::cerr << "The total rs access energy should not be set before, please check analyzer.\n";
  }

  if(reduce_eng.find(config_num) == reduce_eng.end()) {
    reduce_eng[config_num] = totalreduceeng;
  }
  else {
    std::cerr << "The total energy of reduce operation should not be set before, please check analyzer.\n";
  }

  if(dsaccess_eng.find(config_num) == dsaccess_eng.end()) {
    dsaccess_eng[config_num] = totaldneng;
  }
  else {
    std::cerr << "The total ds access energy should not be set before, please check analyzer.\n";
  }

  if(msreg_eng.find(config_num) == msreg_eng.end()) {
    msreg_eng[config_num] = totalmsregeng;
  }
  else {
    std::cerr << "The total register access energy should not be set before, please check analyzer.\n";
  }

  if(msaccess_eng.find(config_num) == msaccess_eng.end()) {
    msaccess_eng[config_num] = totalmslinkeng;
  }
  else {
    std::cerr << "The total register access energy should not be set before, please check analyzer.\n";
  }

  if(multiply_eng.find(config_num) == multiply_eng.end()) {
    multiply_eng[config_num] = totalmultiplyeng;
  }
  else {
    std::cerr << "The total multiply operation energy should not be set before, please check analyzer.\n";
  }
}

unsigned long long MappingStrategy::gettotal_cycle() {
  const unsigned long long cycle = config_cycle[config_num];
  return cycle;
}

double MappingStrategy::gettotal_energy(){
  const double totaleng = spminputread_eng[config_num] + spmweightread_eng[config_num] + spmwrite_eng[config_num] + msreg_eng[config_num]
                  + msaccess_eng[config_num] + multiply_eng[config_num] + reduce_eng[config_num] + rsaccess_eng[config_num]
                  + dsaccess_eng[config_num] + dram_eng;
  return totaleng;
}

double MappingStrategy::getenergy_efficiency() {
  unsigned long long runtime = gettotal_cycle();
  double totaleng = gettotal_energy();
  const double ee = 1 / (runtime * totaleng);
  return ee;
}




