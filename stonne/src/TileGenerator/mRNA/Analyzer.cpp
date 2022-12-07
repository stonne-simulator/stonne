#include "TileGenerator/mRNA/Analyzer.h"
#include <list>

using namespace mRNA;

void Analyzer::parseEnergyPara(std::ifstream& infile) {
  std::string buffer;
  while(getline(infile, buffer)) {
    if(buffer == "}"){
      break;
    }
    std::istringstream record(buffer);
    std::string str;
    while(record >> str) {
      if(isNum(str)) {
        std::cerr << "There should be variables to be set.\n";
      }
      else {
        if (str == "dram_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stod(strnum, &sz);
          energypara->dram = num ;
          break;
        }
        else if(str == "spm_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stod(strnum, &sz);
          energypara->spm = num ;
          break;
        }
        else if(str == "reg_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stoi(strnum, &sz);
          energypara->reg = num;
          break;
        }
        else if(str == "dsaccess_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stoi(strnum, &sz);
          energypara->ds_access = num;
          break;
        }
        else if(str == "msaccess_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stoi(strnum, &sz);
          energypara->ms_access = num;
          break;
        }
        else if(str == "rsaccess_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stoi(strnum, &sz);
          energypara->rs_access = num;
          break;
        }
        else if(str == "multiply_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stoi(strnum, &sz);
          energypara->multiply = num;
          break;
        }
        else if(str == "reduce_eng") {
          std::string strnum = getstr(record);
          std::string::size_type sz;
          double num = std::stoi(strnum, &sz);
          energypara->reduce = num;
          break;
        }
      }
    }
  }
}

void Analyzer::parseconfig(std::ifstream& infile) {
  std::string buffer;
  while(getline(infile, buffer)) {
    std::istringstream record(buffer);
    std::string str;
    while(record >> str) {
      if(isNum(str)) {
        std::cerr << "Syntax error. Check digital position.\n";
      }
      else if(str == "Energy_factors") {
        parseEnergyPara(infile);
      }
      else {
        continue;
      }
    }
  }
}

int Analyzer::CalculateCNNConfig1_Inst(int kernelchannel, int kernelfilters) {
   int instnum = 0;
   if(dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y == 1) {
     instnum = 2 * PUSH_LENGTH + PULL_LENGTH;
   }
   else {
     if(dnn_model->cnn_filter->window_stride != 1) {
       instnum = (kernelchannel + 1) * PUSH_LENGTH + PULL_LENGTH;
     }

     else {
       instnum = (kernelchannel * 2 + 1) * PUSH_LENGTH + PULL_LENGTH;
     }
   }
   return instnum;
}

int Analyzer::CalculateCNNPartial_Inst(int config2vnnum) {
  int instlength = PUSH_LENGTH + PULL_LENGTH;
  return instlength;
}

int Analyzer::CNNConfig1largefil_Inst(int kernel_y, int kernelfilters) {
    int num = 0;
    if(dnn_model->cnn_filter->window_stride == 1) {
        num = (kernelfilters + 2) * kernel_y * PUSH_LENGTH + kernelfilters * PULL_LENGTH;
    }
    else {
      num = (kernelfilters + 1) * kernel_y * PUSH_LENGTH + kernelfilters * PULL_LENGTH;
    }
  return num;
}


double Analyzer::CalculateConfig1_UR(int cycle) {
  double ur = (double(dnn_model->cnn_output->output_x) * double(dnn_model->cnn_output->output_y) * double(dnn_model->cnn_filter->filter_x)
            * double(dnn_model->cnn_filter->filter_y) * double(dnn_model->cnn_filter->filter_channel) * double(dnn_model->cnn_filter->filter_number)) /
            (double(maeri->pe_size) * double(cycle));
  return ur;
}

void Analyzer::CalculateConfig1onchip(MappingStrategy* map, std::vector<double> partial, int type) {
  if(type == 0) {
    double onchipinput = dnn_model->cnn_input->input_x * dnn_model->cnn_input->input_y * map->kernel_c * 2 / 1024;
    double onchipweight = dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y * map->kernel_c * map->kernel_n * 2 / 1024;
    double onchipoutput = dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * map->kernel_n * 2 / 1024;
    double onchipspace = onchipinput + onchipweight + onchipoutput;
    map->setonchip_input(onchipinput);
    map->setonchip_weight(onchipweight);
    map->setonchip_output(onchipoutput);
    map->setonchip_space(onchipspace);
  }
  else if(type == 1) {

  }
  else if(type == 2) {

  }
  else {
    double onchipinput = dnn_model->cnn_input->input_x * dnn_model->cnn_input->input_y * dnn_model->cnn_input->input_channel * 2 / 1024;
    for(std::vector<double>::iterator I = partial.begin(), E = partial.end(); I != E; I++) {
      double part = *I;
      onchipinput += part;
    }
    double onchipweight = dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y * dnn_model->cnn_filter->filter_channel * 2 / 1024;
    double onchipoutput = dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * dnn_model->cnn_output->output_channel * 2 / 1024;
    double onchipspace = onchipinput + onchipoutput + onchipweight;
    map->setonchip_input(onchipinput);
    map->setonchip_weight(onchipweight);
    map->setonchip_output(onchipoutput);
    map->setonchip_space(onchipspace);
  }
}

double Analyzer::CalculatePartialsum(int psnum) {
  double sum = dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * dnn_model->cnn_output->output_channel * psnum * 2 / 1024;
  return sum;
}

void Analyzer::CalculateEnergy(MappingStrategy* map, int config_num) {

  unsigned long long spminputread = map->spminput_read[config_num];
  unsigned long long spmweightread = map->spmweight_read[config_num];
  unsigned long long spmwrite = map->spm_write[config_num];
  unsigned long long dnaccess = map->dn_access[config_num];
  unsigned long long msregaccess = map->msreg_access[config_num];
  unsigned long long msaccess = map->mslink_access[config_num];
  unsigned long long rsaccess = map->rn_access[config_num];
  unsigned long long multiply = map->multiply_num[config_num];
  unsigned long long reduce = map->reduce_num[config_num];

  double spminputreadeng = spminputread * energypara->spm;
  double spmweightreadeng = spmweightread * energypara->spm;
  double spmwriteeng = spmwrite * energypara->spm;
  double msregeng = msregaccess * energypara->reg;
  double mslinkeng = msaccess * energypara->ms_access;
  double multiplyeng = multiply * energypara->multiply;
  double rsaccesseng = rsaccess * energypara->rs_access;
  double reduceeng = reduce * energypara->reduce;
  double dnaccesseng = dnaccess * energypara->ds_access;

  map->setspminputread_energy(config_num, spminputreadeng);
  map->setspmweightread_energy(config_num, spmweightreadeng);
  map->setspmwrite_energy(config_num, spmwriteeng);
  map->setmsreg_energy(config_num, msregeng);
  map->setmslink_energy(config_num, mslinkeng);
  map->setmultiply_energy(config_num, multiplyeng);
  map->setrsaccess_energy(config_num, rsaccesseng);
  map->setreduce_energy(config_num, reduceeng);
  map->setdnaccess_energy(config_num, dnaccesseng);
}

/*
void Analyzer::CalculateCycle(MappingStrategy* map, int config_num) {
  unsigned long long input_trans = 0;
  unsigned long long filter_trans = 0;
  if(map->input_trans.find(config_num) != map->input_trans.end() && map->filter_trans.find(config_num) != map->filter_trans.end()) {
    unsigned long long input_trans = map->input_trans[config_num];
    unsigned long long filter_trans = map->filter_trans[config_num];
  }
  else{
    std::cerr << "The input or weight trans information must be calculated before calculating the cycle, pleaase check the analyzer.\n";
  }
}
 */

void Analyzer::writeProfile(std::ofstream& profile, std::map<int, MappingStrategy*> mappings) {
  profile << "Model Name: " << dnn_model->model_name << "\n";
  profile << "Layer Type: " << dnn_model->layer_type << "\n";
  profile << "Layer Number: " << dnn_model->layer_num << "\n";
  profile << "Total number of Multiplier Switches: " << maeri->pe_size << "\n";
  profile << "Total number of Mapping Strategies: " << mapping_num << "\n\n";
  for (int i = 0; i < mapping_num; i++) {
    profile << "==================================================================\n";
    MappingStrategy *map = mappings[i];
    profile << "Mapping Strategy " << i << ": \n";
    profile << "Tile Size: <" << "T_X = " << map->kernel_x << ", T_Y = " << map->kernel_y
            << ", T_C = " << map->kernel_c << ", T_K = " << map->kernel_n << ", T_N = "
            << map->kernel_in << ", T_X' = " << map->kernel_ox << ", T_Y' = " << map->kernel_oy << ">\n";
    profile << "Average utilization rate: " << map->average_ur << "\n\n";
    int confignum = map->config_num + 1;
    for (int j = 0; j < confignum; j++) {
      profile << "-------------------------------------------------------------------\n";
      if (j == confignum - 1) {
        profile << "Total Configurations: \n\n";
      } else {
        profile << "Configuration " << j << "\n\n";
      }
      profile << "Mapping: ";
      if(j != confignum - 1) {
        int vn_size = map->elementinvn[j];
        int vn_num = map->vn_num[j];
        int idle = maeri->pe_size - vn_num * vn_size;
        profile << "- Size of Virtual Neuron (VN): " << vn_size << "\n";
        profile << "- Number of VNs: " << vn_num << "\n";
        profile << "- Number of Idle Multiplier Switches: " << idle << "\n\n";
      }

      profile << "Distribute Network: \n";
      if(j != confignum - 1) {
        profile << "- Number of unicast in weights: ";
        for (std::vector<int>::iterator I = map->weight_unicast[j].begin(), E = map->weight_unicast[j].end();
             I != E; I++) {
          int unicast = *I;
          profile << "(" << unicast << ") ";
        }
        profile << "\n";
        profile << "- Number of multicast in weights: ";
        for (std::vector<int>::iterator I = map->weight_multicast[j].begin(), E = map->weight_multicast[j].end();
             I != E; I++) {
          int multicast = *I;
          profile << "(" << multicast << ") ";
        }
        profile << "\n";
        profile << "- Size of multicast in weights: ";
        for (std::vector<int>::iterator I = map->weightmulticast_size[j].begin(), E = map->weightmulticast_size[j].end();
             I != E; I++) {
          int size = *I;
          profile << "(" << size << ") ";
        }
        profile << "\n";
        profile << "- Number of unicast in inputs: ";
        for (std::vector<int>::iterator I = map->input_unicast[j].begin(), E = map->input_unicast[j].end();
             I != E; I++) {
          int unicast = *I;
          profile << "(" << unicast << ") ";
        }
        profile << "\n";
        profile << "- Number of multicast in inputs: ";
        for (std::vector<int>::iterator I = map->input_multicast[j].begin(), E = map->input_multicast[j].end();
             I != E; I++) {
          int multicast = *I;
          profile << "(" << multicast << ") ";
        }
        profile << "\n";
        profile << "- Size of unicast in inputs: ";
        for (std::vector<int>::iterator I = map->inputmulticast_size[j].begin(), E = map->inputmulticast_size[j].end();
             I != E; I++) {
          int size = *I;
          profile << "(" << size << ") ";
        }
        profile << "\n";
      }
      unsigned long long dn_access = map->dn_access[j];
      profile << "- Number of access between distribute switches: " << dn_access << "\n";
      if(show_energy) {
        double dn_eng = map->dsaccess_eng[j];
        profile << "- Normalized energy of distribute network " << dn_eng << "\n\n";
      }
      else {
        profile << "\n";
      }

      profile << "Reduce Network: \n";
      unsigned long long reduce_operation = map->reduce_num[j];
      profile << "- Number of reduce operations: " << reduce_operation << "\n";
      unsigned long long  rn_access = map->rn_access[j];
      profile << "- Number of access between reduce switches: " << rn_access << "\n";
      if(show_energy) {
        double rn_access = map->rsaccess_eng[j];
        double reduceeng = map->reduce_eng[j];
        profile << "- Normalized energy of Reduce Network: " << rn_access + reduceeng << " ( accessing RSs = "
                << rn_access << ", reduce operation = " << reduceeng << " )\n\n";
      }
      else{
        profile << "\n";
      }

      profile << "Multiplier Network: \n";
      unsigned long long multiply = map->multiply_num[j];
      profile << "- Number of multiplications: " << multiply << "\n";
      unsigned long long msreg = map->msreg_access[j];
      profile << "- Number of register access of the multiplier switch: " << msreg << "\n";
      unsigned long long ms_link = map->mslink_access[j];
      profile << "- Number of forward access between multiplier switches: " << ms_link << "\n";
      if(show_energy) {
        double mslink = map->msaccess_eng[j];
        double msreg = map->msreg_eng[j];
        double multi = map->multiply_eng[j];
        profile << "- Normalized energy of Multiplier Network: " << mslink+msreg+multi << " ( forward energy = "
                << mslink << ", accessing local buffer in MS = " << msreg << ", multiply operations = " << multi << " )\n\n";
      }
      else {
        profile << "\n";
      }

      profile << "Virtual Neuron Address Table (VNAT): \n";
      if(j != confignum - 1) {
        if (map->config_attribute[j] == conv || map->config_attribute[j] == fc || map->config_attribute[j] == lstm) {
          profile << "- Number of outputs expected (single control step): " << map->vn_num[j] << "\n\n";
        } else {
          profile << "- Number of outputs expected: 0\n\n";
        }
      }

      profile << "On-chip Scratchpad memory (Prefetch buffer): \n";
      unsigned long long weight_read = map->spmweight_read[j];
      unsigned long long input_read = map->spminput_read[j];
      unsigned long long write = map->spm_write[j];
      profile << "- Number of weights read: " << weight_read << "\n";
      profile << "- Number of inputs read: " << input_read << "\n";
      profile << "- Number of outputs write: " << write << "\n";
      profile << "- Number of input data streams: " << map->input_trans[j] << "\n";
      profile << "- Number of weight data streams: " << map->filter_trans[j] << "\n";
      if(show_energy) {
        double spminputread = map->spminputread_eng[j];
        double spmweightread = map->spmweightread_eng[j];
        double spmwrite = map->spmwrite_eng[j];
        profile << "- Normalized energy of accessing on-chip Scratchpad Memory: " << spminputread+spmweightread+spmwrite
                << " ( SPM input read = " << spminputread << ", SPM weight read = " << spmweightread << ", SPM write = " << spmwrite << " )\n\n";
      }
      else {
        profile << "\n";
      }

      profile << "Performance: \n";
      unsigned long long control_step = map->config_cs[j];
      unsigned long long cycle = map->config_cycle[j];
      profile << "- Number of control steps: " << control_step << "\n";
      profile << "- Runtime cycles: " << cycle << "\n";
      if(j != confignum - 1) {
        profile << "- Peak utilization rate in configuration " << j << " is: " << map->peak_ur[j] << "\n";
        profile << "- Average utilization rate in configuration " << j << " is: " << map->config_ur[j] << "\n\n";
      }

      profile << "Code Size: \n";
      double code = map->code_size[j];
      profile << "- Code size is: " << code << "\n\n";

      if(j == confignum - 1) {
        if(show_energy) {
          profile << "- Dram energy is: " << map->dram_eng << "\n";
        }
        profile << "- Average utilization rate throughout the whole layer: " << map->average_ur << "\n\n";
      }
    }
  }
}

bool Analyzer::checkCNNInput(){
  bool correct = false;
  if(dnn_model->cnn_input->input_channel == dnn_model->cnn_filter->filter_channel && dnn_model->cnn_output->output_channel == dnn_model->cnn_filter->filter_number) {
    correct = true;
  }
  return correct;
}

void Analyzer::CalculateDSN(MappingStrategy* map, int confignum, int type) {
  int elementnum = map->elementinvn[confignum];
  int vnnum = map->vn_num[confignum];
  unsigned long long inputtrans = map->input_trans[confignum];
  unsigned long long  filtertrans = map->filter_trans[confignum];

  bool finish = false;
  int level = 0;
  int edge = 0;
  int singleedge = 0;
  int copy = vnnum;
  int penum = maeri->pe_size;
  int outputx = dnn_model->cnn_output->output_x;
  int filtery = dnn_model->cnn_filter->filter_y;
  unsigned long long inputedges = 0;
  unsigned long long filteredges = 0;
  unsigned long long totalds = 0;

  while (!finish) {
    if(penum == 1) {
      break;
    }
    else {
      penum = penum / 2;
    }
    level++;
  }


  int l = 0;
  while(copy != 1) {
    singleedge = singleedge + copy;
    l++;
    if(copy % 2 == 0) {
      copy = copy / 2;
    }
    else {
      copy = std::floor(double(copy) / 2) + 1;
    }
  }
  singleedge = ((level - l) * vnnum + singleedge);
//Type 0: Conv, including filter and edge
  if(type == 0) {
    filteredges = filtertrans * (unsigned long long)(level) * (unsigned long long)(vnnum) * (unsigned long long)(elementnum);

    if(dnn_model->cnn_filter->window_stride == 1) {
      unsigned long long temp1 = ((unsigned long long)edge * (inputtrans - outputx * filtertrans)
                               * (unsigned long long)(elementnum / filtery));
      unsigned long long temp2 = singleedge * outputx * filtertrans * elementnum;
      inputedges =  temp1 + temp2;
    }
    else {
      inputedges = (singleedge * elementnum * inputtrans);
    }
    totalds = inputedges + filteredges;
    map->setdn_access(confignum, totalds);
  }
//Type 2: Partial sum or Max Pool
  else if(type == 1){
    edge = level * (elementnum * vnnum);
    inputedges = edge * inputtrans;
    map->setdn_access(confignum, inputedges);
  }
//Type 3: Hardmard product
  else if(type == 2){
    edge = level * elementnum * vnnum;
    inputedges = 2 * edge * inputtrans;
    map->setdn_access(confignum, inputedges);
  }
//Type 4: Fully connected or RNN gate calculate:
  else {
    edge = level * elementnum * vnnum;
    inputedges = singleedge * elementnum * inputtrans;
    filteredges = edge * filtertrans;
    totalds = inputedges + filteredges;
    map->setdn_access(confignum, totalds);
  }
}

void Analyzer::CalculateMSN(MappingStrategy* map, int confignum, int type){

  int elementnum = map->elementinvn[confignum];
  int vnnum = map->vn_num[confignum];
  unsigned long long inputtrans = map->input_trans[confignum];
  unsigned long long  filtertrans = map->filter_trans[confignum];
  int kernelch = map->kernel_c;

  int outputx = dnn_model->cnn_output->output_x;
  int filterx = dnn_model->cnn_filter->filter_x;
  int filtery = dnn_model->cnn_filter->filter_y;
  unsigned long long filterreg = 0;
  unsigned long long inputreg = 0;
  unsigned long long totalreg = 0;
  unsigned long long forward = 0;
  unsigned long long multiply = 0;
  //Conv
  if(type == 0) {
    if(dnn_model->cnn_filter->window_stride == 1) {
      int singleforward = filterx * (filtery - 1) * vnnum * kernelch;
      forward = (unsigned long long)(singleforward) * (inputtrans - outputx * filtertrans);
      totalreg = elementnum * vnnum * outputx * filtertrans + vnnum * (elementnum / filtery) * (inputtrans - outputx * filtertrans)
              + elementnum * vnnum * inputtrans;
      multiply = elementnum * vnnum * inputtrans;
      map->setmslink_access(confignum, forward);
      map->setmsreg_access(confignum, totalreg);
      map->setmultiply_num(confignum, multiply);
    }
    else {
      inputreg = 2 * elementnum * vnnum * inputtrans;
      totalreg = inputreg + totalreg;
      multiply = elementnum * vnnum * inputtrans;
      map->setmslink_access(confignum, forward);
      map->setmsreg_access(confignum, totalreg);
      map->setmultiply_num(confignum, multiply);
    }
  }
  //Partial sum or Max pool
  else if(type == 1) {
    inputreg = vnnum * elementnum * inputtrans;
    totalreg = inputreg + filterreg;
    map->setmslink_access(confignum, forward);
    map->setmsreg_access(confignum, totalreg);
    map->setmultiply_num(confignum, multiply);
  }
  //Hardmard product
  else if(type == 2){
    inputreg = vnnum * elementnum * inputtrans;
    filterreg = vnnum * elementnum * filtertrans;
    totalreg = inputreg + filterreg;
    multiply = vnnum * elementnum * inputtrans;
    map->setmslink_access(confignum, forward);
    map->setmsreg_access(confignum, totalreg);
    map->setmultiply_num(confignum, multiply);
  }
  //Fully Connected
  else {
    inputreg = elementnum * vnnum * inputtrans;
    filterreg = elementnum * vnnum * filtertrans;
    totalreg = inputreg + filterreg;
    multiply = elementnum * vnnum * filtertrans;
    map->setmslink_access(confignum, forward);
    map->setmsreg_access(confignum, totalreg);
    map->setmultiply_num(confignum, multiply);
  }
}

void Analyzer::CalculateRSN(MappingStrategy* map, int confignum) {

  int elementnum = map->elementinvn[confignum];
  int vnnum = map->vn_num[confignum];
  unsigned long long inputtrans = map->input_trans[confignum];

  int finish = false;
  int penum = maeri->pe_size;
  int level = 0;
  int copy = elementnum;
  int edge = 0;
  int reduce = 0;
  unsigned long long rsnedges = 0;
  unsigned long long rnreduce = 0;

  while (!finish) {
    if(penum == 1) {
      break;
    }
    else {
      penum = penum / 2;
    }
    level++;
  }
  int l = 0;
  while(copy != 1) {
    edge = edge + copy;
    reduce = reduce + (copy / 2);
    l++;
    if(copy % 2 == 0) {
      copy = copy / 2;
    }
    else {
      copy = std::floor(double(copy) / 2) + 1;
    }
  }
  rsnedges = (unsigned long long)(edge * vnnum) * inputtrans;
  rnreduce = (unsigned long long)(reduce * vnnum) * inputtrans;
  map->setrn_access(confignum, rsnedges);
  map->setreduce_num(confignum, rnreduce);
}

void Analyzer::CalculateSPM(MappingStrategy* map, int confignum, int type) {

  int elementnum = map->elementinvn[confignum];
  int vnnum = map->vn_num[confignum];
  unsigned long long inputtrans = map->input_trans[confignum];
  unsigned long long filtertrans = map->filter_trans[confignum];

  int outputx = dnn_model->cnn_output->output_x;
  int filtery = dnn_model->cnn_filter->filter_y;
  //CONV
  if(type == 0) {
    unsigned long long filterdata = elementnum * vnnum * filtertrans;
    unsigned long long spmwrite = (unsigned long long)vnnum * inputtrans;
    unsigned long long inputdata = 0;
    if(dnn_model->cnn_filter->window_stride == 1) {
      inputdata = (unsigned long long)(elementnum * outputx) * filtertrans
                + (unsigned long long)(elementnum / filtery) * (inputtrans - outputx * filtertrans);
    }
    else {
      inputdata = elementnum * (unsigned long long)vnnum * inputtrans;
    }
    map->setspminput_read(confignum, inputdata);
    map->setspmweight_read(confignum, filterdata);
    map->setspm_write(confignum, spmwrite);
  }
  //Partial sum or Max pool
  else if(type == 1) {
    unsigned long long spmread = inputtrans * elementnum * (unsigned long long)vnnum;
    unsigned long long spmwrite = inputtrans * (unsigned long long)vnnum;
    map->setspminput_read(confignum, spmread);
    map->setspmweight_read(confignum, 0);
    map->setspm_write(confignum, spmwrite);
  }
  //Hardmard product
  else if(type == 2) {
    unsigned long long spmread = (unsigned long long)( elementnum * vnnum) * inputtrans;
    unsigned long long spmwrite = (unsigned long long)vnnum * inputtrans;
    map->setspminput_read(confignum, spmread);
    map->setspmweight_read(confignum, spmread);
    map->setspm_write(confignum, spmwrite);
  }
  //Fully connected
  else {
    unsigned long long inputdata = inputtrans * (unsigned long long)elementnum;
    unsigned long long filterdata = filtertrans * (unsigned long long)(elementnum * vnnum);
    unsigned long long spmwrite = (unsigned long long)vnnum * filtertrans;
    map->setspminput_read(confignum, inputdata);
    map->setspmweight_read(confignum, filterdata);
    map->setspm_write(confignum, spmwrite);
  }
}

void Analyzer::CalculateDram(MappingStrategy* map, std::vector<double> partialsum, int type) {
  int inputx = dnn_model->cnn_input->input_x;
  int inputy = dnn_model->cnn_input->input_y;
  int inputc = dnn_model->cnn_input->input_channel;
  int outputx = dnn_model->cnn_output->output_x;
  int outputy = dnn_model->cnn_output->output_y;
  int outputc = dnn_model->cnn_output->output_channel;

  int filterx = dnn_model->cnn_filter->filter_x;
  int filtery = dnn_model->cnn_filter->filter_y;
  int filterc = dnn_model->cnn_filter->filter_channel;
  int filtern = dnn_model->cnn_filter->filter_number;

  if(type == 0) {
    unsigned long long origndatatrans = (unsigned long long)(std::ceil(double((inputx * inputy * inputc + filterx * filtery * filterc * filtern
                                                                               + outputx * outputy * outputc) * 2) / 8));
    int pardata = 0;
    int lastpar = 0;
    for (std::vector<double>::iterator I = partialsum.begin(), E = partialsum.end(); I != E; I++) {
      double partial = *I;
      pardata = pardata + partial;
    }
    //For ach partial, they are written to the main memory and read back to generate the output. Bandwidth is 64bit
    unsigned long long partrans = pardata * 128 * 2;
    unsigned long long totaltrans = origndatatrans + partrans;
    double dram_eng = totaltrans * energypara->dram;
    map->setdram_access(totaltrans);
    map->setdram_energy(dram_eng);
  }
}

bool Analyzer::compMappingcycle(MappingStrategy*& m1, MappingStrategy*& m2) {
  const unsigned long long c1 = m1->gettotal_cycle();
  const unsigned long long c2 = m2->gettotal_cycle();
  return c1 < c2;
}

bool Analyzer::compMappingEnergy(MappingStrategy* m1, MappingStrategy* m2) {
  const double e1 = m1->gettotal_energy();
  const double e2 = m2->gettotal_energy();
  return e1 < e2;
}

bool Analyzer::compMappingEE(MappingStrategy* m1, MappingStrategy* m2) {
  const double ee1 = m1->getenergy_efficiency();
  const double ee2 = m2->getenergy_efficiency();
  return ee1 > ee2;
}

MappingStrategy* Analyzer::SortMappingStrategy(OptGoal goal) {
  std::list<MappingStrategy* > sortlist;
  for(std::map<int, MappingStrategy* >::iterator I = mappings.begin(), E = mappings.end(); I != E; I++) {
    MappingStrategy* temp = I->second;
    sortlist.push_back(temp);
  }

  if(goal == performance) {
      sortlist.sort(Analyzer::compMappingcycle);
  }
  else if(goal == energy) {
    sortlist.sort(Analyzer::compMappingEnergy);
  }
  else if(goal == energy_efficiency){
    sortlist.sort(Analyzer::compMappingEE);
  }

  mappings.clear();
  int i = 0;
  std::list<MappingStrategy* >::iterator list_iter = sortlist.begin();
  while (list_iter != sortlist.end()) {
    mappings[i] = *list_iter;
    i++;
    list_iter++;
  }

  return mappings[0];
}

void Analyzer::AnalyzeCNN(std::ofstream& profile, OptGoal option) {
   if(!checkCNNInput()) {
     std::cerr << "Parameter of CNN is not correct, please check again.\n";
   }

  int filter_xy = dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y;
  int filter_xyc = filter_xy * dnn_model->cnn_filter->filter_channel;
  int filter_xycn = filter_xyc * dnn_model->cnn_filter->filter_number;
  std::vector<int> KernelChannel;
  int cmulf = 0;
  int map_num = 0;

//If the total mul and add number less than PE number, we use this data flow because there is no partial sum at all
//And there is no reconfiguration, only configuration 1.
  int testnum = std::floor(double(maeri->pe_size) / double(filter_xyc));
  double testur = double(filter_xyc * testnum) / double(maeri->pe_size);
  if((filter_xyc <= maeri->pe_size && testur >= 0.8) || filter_xycn < maeri->pe_size) {
    MappingStrategy* map = new MappingStrategy();
    int config_num = 0;
    map->setconfig_num(config_num+1);
    map->setconfig_attribute(config_num, conv);
    map->setkernel_x(dnn_model->cnn_filter->filter_x);
    map->setkernel_y(dnn_model->cnn_filter->filter_y);
    map->setkernel_c(dnn_model->cnn_filter->filter_channel);

    std::vector<double> partial;
    int config1cycle = 0;
    int config1rc = 0;
    int kernelch = dnn_model->cnn_input->input_channel;
    int kernelfil = std::floor(double(maeri->pe_size) / double(filter_xyc));
    if(dnn_model->cnn_filter->filter_number * filter_xyc < maeri->pe_size) {
      kernelfil = dnn_model->cnn_filter->filter_number;
    }
    int weightunicast = kernelfil * filter_xyc;
    double peakur = double(weightunicast) / double(maeri->pe_size);
    int secondunicast = dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_channel;
    map->setinput_unicast(config_num, filter_xyc);
    map->setinput_unicast(config_num, secondunicast);
    map->setweight_unicast(config_num, weightunicast);
    map->setweight_multicast(config_num, 0);
    map->setinput_multicast(config_num, filter_xyc);
    map->setweightmulticast_size(config_num, 0);
    map->setinputmulticast_size(config_num, kernelfil);
    map->setpeak_ur(config_num, peakur);
    map->setkernel_n(kernelfil);
    map->setvn_num(config_num, kernelfil);
    map->setelementinvn(config_num, filter_xyc);
    int config1filtrans = std::ceil(double(dnn_model->cnn_filter->filter_number) / double(kernelfil));
    map->setfilter_trans(config_num, config1filtrans);
    config1cycle = (unsigned long long)(dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * config1filtrans);
    map->setcontrol_step(config_num, config1cycle);
    map->setinput_trans(config_num, config1cycle);
    if(dnn_model->cnn_filter->window_stride == 1) {
      config1rc = (config1cycle - dnn_model->cnn_output->output_x * config1filtrans) *
              std::ceil(double(dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_channel) / double(maeri->dsnet->bandwidth))
              + (dnn_model->cnn_output->output_x * config1filtrans) * std::ceil(double(filter_xyc) / double(maeri->dsnet->bandwidth))
              + config1filtrans * std::ceil(double(kernelfil * filter_xyc) / double(maeri->dsnet->bandwidth));
      map->setcycle(config_num, config1rc);
    }
    else {
      config1rc = config1cycle * std::ceil(double(filter_xyc) / double(maeri->dsnet->bandwidth))
                + config1filtrans * std::ceil(double(kernelfil * filter_xyc) / double(maeri->dsnet->bandwidth));
      map->setcycle(config_num, config1rc);
    }
    CalculateDSN(map, config_num, 0);
    CalculateMSN(map, config_num, 0);
    CalculateRSN(map, config_num);
    CalculateSPM(map, config_num, 0);
    CalculateEnergy(map, config_num);
    int config1_inst_num = CalculateCNNConfig1_Inst(kernelch, kernelfil);
    map->setcode_size(config_num, config1_inst_num);
    double config1ur = CalculateConfig1_UR(config1cycle);
    map->setconfig_ur(config_num, config1ur);
    map->setaverage_ur(config1ur);
    CalculateConfig1onchip(map, partial, 1);
    partial.push_back(0);
    CalculateDram(map, partial, 0);
    map->sum_all();
    mappings[map_num] = map;
    map_num++;
  }

//If the 2D filter size is small, but there must be reconfiguration, we using the weight stationary is better.
  else if (filter_xy <= maeri->pe_size) {
//Calculate all the possible combination of the channel and filters that can be put on the maeri.
    int pesize = maeri->pe_size;
    cmulf = std::floor(double(pesize) / double(filter_xy));
    cmulf++;
    while(KernelChannel.empty()) {
      cmulf--;
      for (int n = 1; n * n <= cmulf; n++) {
        if (cmulf % n == 0) {
          int temp = cmulf / n;
          if (n <= dnn_model->cnn_filter->filter_channel && temp <= dnn_model->cnn_filter->filter_number) {
            KernelChannel.push_back(n);
          }
          if(temp <= dnn_model->cnn_filter->filter_channel && n <= dnn_model->cnn_filter->filter_number) {
            KernelChannel.push_back(temp);
          }
          continue;
        }
      }
    }

    for (std::vector<int>::iterator I = KernelChannel.begin(); I != KernelChannel.end(); I++) {
      MappingStrategy* map = new MappingStrategy();
      int config_num = 0;
      map->setconfig_attribute(config_num, conv);
      unsigned long long config1cycle = 0;
      unsigned long long config2cycle = 0;
      unsigned long long config3cycle = 0;
      double config1ur = 0;
      double config2ur = 0;
      double config3ur = 0;

      unsigned long long config1rc = 0;
      unsigned long long config2rc = 0;
      unsigned long long config3rc = 0;
      std::vector<std::pair<int, double> > info;
      std::vector<double> partial;

      int kernelch = *I;
      int kernelfil = cmulf / kernelch;
      int singlevn = filter_xy * kernelch;
      int kernelelementnum = cmulf * filter_xy;
      double peak_ur = double(kernelelementnum) / double(maeri->pe_size);
      int secondunicast = dnn_model->cnn_filter->filter_x * kernelch;
      map->setkernel_x(dnn_model->cnn_filter->filter_x);
      map->setkernel_y(dnn_model->cnn_filter->filter_y);
      map->setkernel_c(kernelch);
      map->setkernel_n(kernelfil);
      map->setelementinvn(config_num, singlevn);
      map->setvn_num(config_num, kernelfil);
      map->setinput_unicast(config_num, singlevn);
      map->setinput_unicast(config_num, secondunicast);
      map->setinput_multicast(config_num, singlevn);
      map->setweight_unicast(config_num, kernelelementnum);
      map->setweight_multicast(config_num, 0);
      map->setweightmulticast_size(config_num, 0);
      map->setinputmulticast_size(config_num, kernelfil);
      map->setpeak_ur(config_num, peak_ur);
      int config1_inst_num = CalculateCNNConfig1_Inst(kernelch, kernelfil);
      map->setcode_size(config_num, config1_inst_num);
      int vnpartialsum = std::ceil(double(dnn_model->cnn_input->input_channel) / double(kernelch));
      int config1filtrans = std::ceil(double(dnn_model->cnn_filter->filter_number) / double(kernelfil)) * vnpartialsum;
      map->setfilter_trans(config_num, config1filtrans);
      config1cycle = dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * config1filtrans;
      map->setcontrol_step(config_num, config1cycle);
      map->setinput_trans(config_num, config1cycle);
      if(dnn_model->cnn_filter->window_stride == 1) {
        config1rc = (config1cycle - dnn_model->cnn_output->output_x * config1filtrans) *
            std::ceil(double(dnn_model->cnn_filter->filter_x * kernelch) / double(maeri->dsnet->bandwidth))
            + (dnn_model->cnn_output->output_x * config1filtrans) * std::ceil(double(singlevn) / double(maeri->dsnet->bandwidth))
            + config1filtrans * std::ceil(double(kernelelementnum) / double(maeri->dsnet->bandwidth));
      }
      else {
        config1rc = config1cycle * std::ceil(double(singlevn) / double(maeri->dsnet->bandwidth))
                  + config1filtrans * std::ceil(double(kernelelementnum) / double(maeri->dsnet->bandwidth));
      }
      map->setcycle(config_num, config1rc);
      CalculateDSN(map, config_num, 0);
      CalculateMSN(map, config_num, 0);
      CalculateRSN(map, config_num);
      CalculateSPM(map, config_num, 0);
      CalculateEnergy(map, config_num);
      config1ur = CalculateConfig1_UR(config1cycle);
      map->setconfig_ur(config_num, config1ur);
      CalculateConfig1onchip(map, partial, 1);
      double psafterconfig1 = CalculatePartialsum(vnpartialsum);
      partial.push_back(psafterconfig1);
      if(vnpartialsum <= maeri->pe_size) {
        config_num++;
        map->setconfig_num(config_num+1);
        map->setconfig_attribute(config_num, ps);
        int config2vnnum = std::floor(maeri->pe_size / vnpartialsum);
        int unicast = config2vnnum * vnpartialsum;
        map->setvn_num(config_num, config2vnnum);
        map->setelementinvn(config_num, vnpartialsum);
        map->setinput_unicast(config_num, unicast);
        map->setweight_unicast(config_num, 0);
        map->setweight_multicast(config_num, 0);
        map->setinput_multicast(config_num, 0);
        map->setweightmulticast_size(config_num, 0);
        map->setinputmulticast_size(config_num, 0);
        double peak_ur = config2vnnum * vnpartialsum / maeri->pe_size;
        map->setpeak_ur(config_num, peak_ur);
        int config2_inst_num = CalculateCNNPartial_Inst(config2vnnum);
        map->setcode_size(config_num, config2_inst_num);
        config2cycle = std::ceil(double(dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * dnn_model->cnn_output->output_channel) / double(config2vnnum));
        map->setcontrol_step(config_num, config2cycle);
        map->setinput_trans(config_num, config2cycle);
        map->setfilter_trans(config_num, 0);
        config2rc = config2cycle * std::ceil(double(unicast) / double(maeri->dsnet->bandwidth));
        map->setcycle(config_num, config2rc);
        CalculateDSN(map, config_num, 1);
        CalculateMSN(map, config_num, 1);
        CalculateRSN(map, config_num);
        CalculateSPM(map, config_num, 1);
        CalculateEnergy(map, config_num);
        config2ur = double(dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * dnn_model->cnn_output->output_channel * vnpartialsum)
                  / double(maeri->pe_size * config2cycle);
        map->setconfig_ur(config_num, config2ur);
        int total_inst_num = config1_inst_num + config2_inst_num;
        double av_ur = (config1cycle * config1ur + config2cycle * config2ur) / double(config1cycle + config2cycle);
        map->setaverage_ur(av_ur);
        std::pair<int, double>  totalinfo(total_inst_num, av_ur);
        info.push_back(totalinfo);
        CalculateDram(map, partial, 0);
        map->sum_all();
        mappings[map_num] = map;
        map_num++;
      }
      else {
        config_num++;
        map->setconfig_attribute(config_num, ps);
        int config2ps = std::ceil(double(vnpartialsum) / double(maeri->pe_size));
        if(config2ps > maeri->pe_size) {
          delete map;
          continue;
        }
        map->setvn_num(config_num, 1);
        map->setelementinvn(config_num, maeri->pe_size);
        map->setinput_unicast(config_num, maeri->pe_size);
        map->setweight_unicast(config_num, 0);
        map->setweight_multicast(config_num, 0);
        map->setinput_multicast(config_num, 0);
        map->setweightmulticast_size(config_num, 0);
        map->setinputmulticast_size(config_num, 0);
        map->setpeak_ur(config_num, 1);
        config2cycle = dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * dnn_model->cnn_output->output_channel * config2ps;
        map->setcontrol_step(config_num, config2cycle);
        map->setinput_trans(config_num, config2cycle);
        map->setfilter_trans(config_num, 0);
        config2rc = config2cycle * std::ceil(double(maeri->pe_size) / double(maeri->dsnet->bandwidth));
        map->setcycle(config_num, config2rc);
        CalculateDSN(map, config_num, 1);
        CalculateMSN(map, config_num, 1);
        CalculateRSN(map, config_num);
        CalculateSPM(map, config_num, 1);
        CalculateEnergy(map, config_num);
        config2ur = double(vnpartialsum) / double(maeri->pe_size * config2ps);
        map->setconfig_ur(config_num, config2ur);
        int config2_inst_num = CalculateCNNPartial_Inst(1);
        map->setcode_size(config_num, config2_inst_num);
        double psafterconfig2 = CalculatePartialsum(config2ps);
        partial.push_back(psafterconfig2);

        config_num++;
        map->setconfig_num(config_num+1);
        map->setconfig_attribute(config_num, ps);
        int config3vnnum = std::floor(double(maeri->pe_size) / double(config2ps));
        int unicast = config3vnnum * config2ps;
        double peak_ur = double(unicast) / double(maeri->pe_size);
        map->setvn_num(config_num, config3vnnum);
        map->setelementinvn(config_num, config2ps);
        map->setinput_unicast(config_num, unicast);
        map->setweight_unicast(config_num, 0);
        map->setweight_multicast(config_num, 0);
        map->setinput_multicast(config_num, 0);
        map->setweightmulticast_size(config_num, 0);
        map->setinputmulticast_size(config_num, 0);
        map->setpeak_ur(config_num, peak_ur);
        config3cycle = std::ceil(double(dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * dnn_model->cnn_output->output_channel) / double(config3vnnum));
        map->setcontrol_step(config_num, config3cycle);
        map->setinput_trans(config_num, config3cycle);
        map->setfilter_trans(config_num, 0);
        config3rc = config3cycle * std::ceil(double(config2ps * config3vnnum) / double(maeri->dsnet->bandwidth));
        map->setcycle(config_num, config3rc);
        CalculateDSN(map, config_num, 1);
        CalculateMSN(map, config_num,1);
        CalculateRSN(map, config_num);
        CalculateSPM(map, config_num, 1);
        CalculateEnergy(map, config_num);
        config3ur =  double(dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y * dnn_model->cnn_output->output_channel * config2ps)
                  / double(maeri->pe_size * config3cycle);
        map->setconfig_ur(config_num, config3ur);
        int config3_inst_num = CalculateCNNPartial_Inst(config3vnnum);
        map->setcode_size(config_num, config3_inst_num);
        double av_ur = (config1cycle * config1ur + config2cycle * config2ur + config3cycle * config3ur) / double(config1cycle + config2cycle + config3cycle);
        map->setaverage_ur(av_ur);
        CalculateDram(map, partial, 0);
        map->sum_all();
        mappings[map_num] = map;
        map_num++;
      }
    }
  }

//If the 2D filter size is larger than PE number, we must use another dataflow
  else {
    int filter_xc = dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_channel;
    if(filter_xc > maeri->pe_size) {
     std::cerr << "Currently, we don't support CNN layers with large filter size and large channel size.\n";
    }
    int yandf = std::floor(double(maeri->pe_size) / double(filter_xc));
    for(int n = 1; n*n <= yandf; n++) {
      if(yandf % n == 0) {
        int temp = yandf / n;
        KernelChannel.push_back(n);
        KernelChannel.push_back(temp);
      }
    }

    for(std::vector<int>::iterator I = KernelChannel.begin(); I != KernelChannel.end(); I++) {
      int kernel_y = *I;
      int kernel_fil = yandf / kernel_y;
      int num_y = dnn_model->cnn_filter->filter_y / kernel_y;
      int config1_inst_num = CNNConfig1largefil_Inst(num_y, kernel_fil);
      int config1cycle = (dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y);
      int config1ur = 0;
    }
  }
  setmapping_num(map_num);
  bestmap = SortMappingStrategy(option);
  writeProfile(profile, mappings);
}

int Analyzer::CalculateFCConfig1_Inst(int vn_num) {
  int inst_num = 0;
  if(vn_num > 1) {
    inst_num = 2*PUSH_LENGTH + PULL_LENGTH;
  }
  else {
    inst_num = 2 * PUSH_LENGTH + PULL_LENGTH;
  }
  return inst_num;
}

int Analyzer::CalculateFCConfig2_Inst(int vn_num) {
  int inst_num = PUSH_LENGTH + PULL_LENGTH;
  return inst_num;
}

bool Analyzer::checkFCInput(){
  bool correct = false;
  if(dnn_model->cnn_input->input_x == dnn_model->cnn_filter->filter_y && dnn_model->cnn_input->input_y == 1) {
    correct = true;
  }
  return correct;
}

void Analyzer::AnalyzeFC(std::ofstream& profile, OptGoal option) {

  if(!checkFCInput()) {
    std::cerr << "Parameter of the input and weight of the FC Layer doesn't match. Please again!" << "\n";
  }
  std::vector<double> partial;
  MappingStrategy* map = new MappingStrategy();
  int config_num = 0;
  map->setconfig_attribute(config_num, fc);
  int map_num = 0;
  if(dnn_model->cnn_input->input_x <= maeri->pe_size) {
    map->setconfig_num(config_num+1);

    // *** Modified for Stonne
    //int vn_num = std::floor(double(maeri->pe_size) / double(dnn_model->cnn_input->input_x));
    // It's calculating wrong the number of virtual neurons. It only considered input neurons but not output neurons.
    // So, it could happen with ms_num=256 input=16 and output=8: vn_num=ms_num/input=256/16=16 => vn_num > output neurons !!!
    int vn_num = std::min(std::floor(double(maeri->pe_size) / double(dnn_model->cnn_input->input_x)), (double) dnn_model->cnn_output->output_x);

    int maxpe = vn_num * dnn_model->cnn_input->input_x;
    double peak_ur = double(maxpe) / double(maeri->pe_size);
    map->setkernel_x(dnn_model->cnn_input->input_x);
    map->setkernel_y(vn_num);
    map->setkernel_c(1);
    map->setkernel_n(1);
    map->setvn_num(config_num, vn_num);
    map->setelementinvn(config_num, dnn_model->cnn_input->input_x);
    map->setinput_unicast(config_num, dnn_model->cnn_input->input_x);
    map->setweight_unicast(config_num, maxpe);
    map->setweight_multicast(config_num, 0);
    map->setinput_multicast(config_num, dnn_model->cnn_input->input_x);
    map->setweightmulticast_size(config_num, 0);
    map->setinputmulticast_size(config_num, vn_num);
    map->setpeak_ur(config_num, peak_ur);
    int config1_inst = CalculateFCConfig1_Inst(vn_num);
    map->setcode_size(config_num, config1_inst);
    unsigned long long config1_cycle = (unsigned long long)(std::ceil(double(dnn_model->cnn_filter->filter_x) / double(vn_num)));
    unsigned long long config1_rc = (unsigned long long)(std::ceil(double(dnn_model->cnn_input->input_x) / (maeri->dsnet->bandwidth)) * 1)
                                 + (unsigned long long)(std::ceil(double(maxpe) / double(maeri->dsnet->bandwidth)) * config1_cycle);
    map->setcontrol_step(config_num, config1_cycle);
    map->setinput_trans(config_num, 1);
    map->setfilter_trans(config_num, config1_cycle);
    map->setcycle(config_num, config1_rc);
    double config1_ur = double(dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y) / double(maeri->pe_size * config1_cycle);
    map->setconfig_ur(config_num, config1_ur);
    CalculateConfig1onchip(map, partial, 3);
    CalculateDSN(map, config_num, 3);
    CalculateMSN(map, config_num,3);
    CalculateRSN(map, config_num);
    CalculateSPM(map, config_num, 3);
    CalculateEnergy(map, config_num);
    CalculateDram(map, partial, 0);
    map->setaverage_ur(config1_ur);
  }
  else {
    int inputseg = std::ceil(double(dnn_model->cnn_input->input_x) / double(maeri->pe_size));
    map->setkernel_x(maeri->pe_size);
    map->setkernel_y(1);
    map->setkernel_c(1);
    map->setkernel_n(1);
    map->setvn_num(config_num, 1);
    map->setelementinvn(config_num, maeri->pe_size);
    map->setinput_unicast(config_num, maeri->pe_size);
    map->setweight_unicast(config_num, maeri->pe_size);
    map->setweight_multicast(config_num, 0);
    map->setinput_multicast(config_num, 0);
    map->setweightmulticast_size(config_num, 0);
    map->setinputmulticast_size(config_num, 0);
    map->setpeak_ur(config_num, 1);
    int config1_inst = CalculateFCConfig1_Inst(1);
    map->setcode_size(config_num, config1_inst);
    unsigned long long  config1_cycle = (unsigned long long)(dnn_model->cnn_filter->filter_x * inputseg);
    unsigned long long config1_rc = (unsigned long long)(std::ceil(double(maeri->pe_size) / double(maeri->dsnet->bandwidth)) * (inputseg + config1_cycle));
    map->setcontrol_step(config_num, config1_cycle);
    map->setinput_trans(config_num, inputseg);
    map->setfilter_trans(config_num, config1_cycle);
    map->setcycle(config_num, config1_rc);
    double config1_ur = double(dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y) / double(maeri->pe_size * config1_cycle);
    map->setconfig_ur(config_num, config1_ur);
    double partialsum = double(dnn_model->cnn_filter->filter_x * inputseg * 2) / double(1024);
    partial.push_back(partialsum);
    CalculateConfig1onchip(map, partial, 3);
    CalculateDSN(map, config_num, 3);
    CalculateMSN(map, config_num,3);
    CalculateRSN(map, config_num);
    CalculateSPM(map, config_num, 3);
    CalculateEnergy(map, config_num);
    if (inputseg <= maeri->pe_size) {
      config_num++;
      map->setconfig_num(config_num+1);
      map->setconfig_attribute(config_num, ps);
      int vn_num = std::floor(double(maeri->pe_size) / double(inputseg));
      if(dnn_model->cnn_filter->filter_x * inputseg < maeri->pe_size) {
        vn_num = dnn_model->cnn_filter->filter_x;
      }
      map->setvn_num(config_num, vn_num);
      map->setelementinvn(config_num, inputseg);
      int unicast = vn_num * inputseg;
      double peak_ur = double(unicast) / double(maeri->pe_size);
      map->setinput_unicast(config_num, unicast);
      map->setweight_unicast(config_num, 0);
      map->setweight_multicast(config_num, 0);
      map->setinput_multicast(config_num, 0);
      map->setweightmulticast_size(config_num, 0);
      map->setinputmulticast_size(config_num, 0);
      map->setpeak_ur(config_num, peak_ur);
      int config2_inst = CalculateFCConfig2_Inst(vn_num);
      map->setcode_size(config_num, config2_inst);
      int config2_cycle = std::ceil(double(dnn_model->cnn_filter->filter_x) / double(vn_num));
      unsigned long long config2_rc = (unsigned long long)(config2_cycle * std::ceil(double(unicast) / double(maeri->dsnet->bandwidth)));
      map->setcontrol_step(config_num, config2_cycle);
      map->setinput_trans(config_num, config2_cycle);
      map->setfilter_trans(config_num, 0);
      map->setcycle(config_num, config2_rc);
      double config2_ur = double(dnn_model->cnn_filter->filter_x * inputseg) / double(config2_cycle * maeri->pe_size);
      map->setconfig_ur(config_num, config2_ur);
      CalculateDSN(map, config_num, 1);
      CalculateMSN(map, config_num,1);
      CalculateRSN(map, config_num);
      CalculateSPM(map, config_num, 1);
      CalculateEnergy(map, config_num);
      double total_ur = (config1_cycle * config1_ur + config2_cycle * config2_ur) / double(config1_cycle + config2_cycle);
      map->setaverage_ur(total_ur);
    }
    else {
      std::cerr << "Currently, we don't support input size larger than square of the pe_size for fully connected layer ! ";
    }
  }
  map->setconfig_num(config_num+1);
  map->sum_all();
  CalculateDram(map, partial, 0);
  mappings[map_num] = map;
  map_num++;
  setmapping_num(map_num);
  writeProfile(profile, mappings);
}

bool Analyzer::checkRNNInput() {
  bool correct = false;
  if (dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x == dnn_model->cnn_filter->filter_y && dnn_model->cnn_filter->filter_x == 4 * dnn_model->dnn_hidden->hidden_x ) {
    correct = true;
  }
  return correct;
}

int Analyzer::CalculateRNNConfig1_Inst(int vn_num) {
  int inst_num = 0;
  if(vn_num > 1) {
    inst_num = 2 * PUSH_LENGTH + PULL_LENGTH * 2;
  }
  else {
    inst_num = 2 * PUSH_LENGTH + PULL_LENGTH * 2;
  }
  return inst_num;
}

int Analyzer::CalculateRNNConfig2_Inst(int vn_num) {
  int inst_num = PUSH_LENGTH * 2 + 2 * PULL_LENGTH ;
  return inst_num;
}

int Analyzer::CalculateLSTMCt_Inst(int stage, int vn_num) {
  int inst_num = 0;
  if (stage == 1) {
    inst_num = PUSH_LENGTH * 2 + PULL_LENGTH;
  }
  else {
    inst_num = PUSH_LENGTH + PULL_LENGTH+ PUSH_LENGTH * 2 + PULL_LENGTH;
  }
  return inst_num;
}


void Analyzer::AnalyzeRNN(std::ofstream& profile, OptGoal option) {

  if(!checkRNNInput()) {
    std::cerr << "Paramter of the RNN input is not correct. Please check again !" << "\n";
  }
  MappingStrategy* map = new MappingStrategy();
  int config_num = 0;
  int map_num = 0;
  map->setconfig_attribute(config_num, lstm);
  std::vector<std::pair<int, double> > info;
  std::vector<double> partial;
  std::vector<int> cycleinfo;
  int config1_inst = 0;
  unsigned long long config1_cycle = 0;
  double config1_ur = 0;
  int config2_inst = 0;
  unsigned long long config2_cycle = 0;
  double config2_ur = 0;
  int config3_inst = 0;
  unsigned long long config3_cycle = 0;
  double config3_ur = 0;
  int config4_inst = 0;
  unsigned long long config4_cycle = 0;
  double config4_ur = 0;
  int total_inst = 0;
  double total_ur = 0;
  unsigned long long config1_rc = 0;
  unsigned long long config2_rc = 0;
  unsigned long long config3_rc = 0;
  unsigned long long config4_rc = 0;

  //The first is stage is the matrix and vector multiplication.
  if((dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x) <= maeri->pe_size) {
    int elementinvn = dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x;
    int vn_num = std::floor(double(maeri->pe_size) / double(elementinvn));
    int totalelement = vn_num * (elementinvn);
    double peak_ur = double(totalelement) / double(maeri->pe_size);
    map->setkernel_x(dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x);
    map->setkernel_y(vn_num);
    map->setkernel_c(1);
    map->setkernel_n(1);
    map->setvn_num(config_num, vn_num);
    map->setelementinvn(config_num, dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x);
    map->setinput_unicast(config_num, dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x);
    map->setweight_unicast(config_num, totalelement);
    map->setweight_multicast(config_num, 0);
    map->setinput_multicast(config_num, dnn_model->cnn_input->input_x+dnn_model->dnn_hidden->hidden_x);
    map->setweightmulticast_size(config_num, 0);
    map->setinputmulticast_size(config_num, vn_num);
    map->setpeak_ur(config_num, peak_ur);
    config1_inst = CalculateRNNConfig1_Inst(vn_num);
    map->setcode_size(config_num, config1_inst);
    //different activation function
    config1_cycle = std::ceil(3 * double(dnn_model->cnn_filter->filter_x) / (4 * double(vn_num)))
                  + std::ceil(double(dnn_model->cnn_filter->filter_x) / (4 * double(vn_num)));
    config1_rc = config1_cycle * std::ceil(double(totalelement) / double(maeri->dn_bw)) + std::ceil(double(elementinvn) / double(maeri->dn_bw));
    map->setcontrol_step(config_num, config1_cycle);
    map->setinput_trans(config_num, 1);
    map->setfilter_trans(config_num, config1_cycle);
    map->setcycle(config_num, config1_rc);
    config1_ur = double(dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y) / double(maeri->pe_size * config1_cycle);
    map->setconfig_ur(config_num, config1_ur);
    std::pair<int, double> config1info(config1_inst, config1_ur);
    info.push_back(config1info);
    partial.push_back(0);
    CalculateConfig1onchip(map, partial, 3);
    CalculateDSN(map, config_num, 3);
    CalculateMSN(map, config_num,3);
    CalculateRSN(map, config_num);
    CalculateSPM(map, config_num, 3);
    CalculateEnergy(map, config_num);
  }
  else {
    int inputseg = std::ceil(double(dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x) / double(maeri->pe_size));
    map->setkernel_x(maeri->pe_size);
    map->setkernel_y(1);
    map->setkernel_c(1);
    map->setkernel_n(1);
    map->setvn_num(config_num, 1);
    map->setelementinvn(config_num, maeri->pe_size);
    map->setinput_unicast(config_num, maeri->pe_size);
    map->setweight_unicast(config_num, maeri->pe_size);
    map->setweight_multicast(config_num, 0);
    map->setinput_multicast(config_num, 0);
    map->setweightmulticast_size(config_num, 0);
    map->setinputmulticast_size(config_num, 0);
    map->setpeak_ur(config_num, 1);
    //It shoulde be noticed that we use CalculateFCConfig1_Inst instead of CalculateRNNConfig1_Inst.
    config1_inst = CalculateFCConfig1_Inst(1);
    map->setcode_size(config_num, config1_inst);
    config1_cycle = dnn_model->cnn_filter->filter_x * inputseg;
    config1_rc = (config1_cycle + inputseg) * (unsigned long long)(std::ceil(double(maeri->pe_size) / double(maeri->dn_bw)));
    map->setcontrol_step(config_num, config1_cycle);
    map->setinput_trans(config_num, inputseg);
    map->setfilter_trans(config_num, config1_cycle);
    map->setcycle(config_num, config1_rc);
    config1_ur = double(dnn_model->cnn_filter->filter_x * dnn_model->cnn_filter->filter_y) / double(maeri->pe_size * config1_cycle);
    map->setconfig_ur(config_num, config1_ur);
    double partialsum = double(dnn_model->cnn_filter->filter_x * inputseg * 2) / double(1024);
    partial.push_back(partialsum);
    CalculateConfig1onchip(map, partial, 3);
    CalculateDSN(map, config_num, 3);
    CalculateMSN(map, config_num,3);
    CalculateRSN(map, config_num);
    CalculateSPM(map, config_num, 3);
    CalculateEnergy(map, config_num);
    if (inputseg <= maeri->pe_size) {
      config_num++;
      int vn_num = std::floor(double(maeri->pe_size) / double(inputseg));
      if(dnn_model->cnn_filter->filter_x * inputseg < maeri->pe_size) {
        vn_num = dnn_model->cnn_filter->filter_x;
      }
      int totalelement = vn_num * inputseg;
      double peak_ur = double(totalelement) / double(maeri->pe_size);
      map->setvn_num(config_num, vn_num);
      map->setelementinvn(config_num, inputseg);
      map->setinput_unicast(config_num, totalelement);
      map->setweight_unicast(config_num, 0);
      map->setweight_multicast(config_num, 0);
      map->setinput_multicast(config_num, 0);
      map->setweightmulticast_size(config_num, 0);
      map->setinputmulticast_size(config_num, 0);
      map->setpeak_ur(config_num, peak_ur);
      config2_inst = CalculateRNNConfig2_Inst(vn_num);
      map->setcode_size(config_num, config2_inst);
      config2_cycle = std::ceil(double(dnn_model->cnn_filter->filter_x) / double(vn_num));
      config2_rc = config2_cycle * std::ceil(double(totalelement) / double(maeri->dn_bw));
      map->setcontrol_step(config_num, config2_cycle);
      map->setinput_trans(config_num, config2_cycle);
      map->setfilter_trans(config_num, 0);
      map->setcycle(config_num, config2_rc);
      config2_ur = double(dnn_model->cnn_filter->filter_x * inputseg) / double(config2_cycle * maeri->pe_size);
      map->setconfig_ur(config_num, config2_ur);
      CalculateDSN(map, config_num, 2);
      CalculateMSN(map, config_num,2);
      CalculateRSN(map, config_num);
      CalculateSPM(map, config_num, 2);
      CalculateEnergy(map, config_num);
    }
    else {
      std::cerr << "Currently, we don't support input size larger than 65536 !\n";
    }
  }

  //The second stage is to calculate the hadamard product which produces the h_t and c_t
  config_num++;
  int vn_number = 0;
  if(2 * dnn_model->dnn_hidden->hidden_x > maeri->pe_size) {
    vn_number = maeri->pe_size / 2;
  }
  else {
    vn_number = dnn_model->dnn_hidden->hidden_x;
  }
  double peak_ur = double(2 * vn_number) / double(maeri->pe_size);
  map->setvn_num(config_num, vn_number);
  map->setelementinvn(config_num, 2);
  map->setinput_unicast(config_num, 2 * vn_number);
  map->setweight_unicast(config_num, 2 * vn_number);
  map->setweight_multicast(config_num, 0);
  map->setinput_multicast(config_num, 0);
  map->setweightmulticast_size(config_num, 0);
  map->setinputmulticast_size(config_num, 0);
  map->setpeak_ur(config_num, peak_ur);
  config3_inst = CalculateLSTMCt_Inst(1, 1);
  map->setcode_size(config_num, config3_inst);
  config3_cycle = std::ceil(double(dnn_model->dnn_hidden->hidden_x * 2) / double(maeri->pe_size));
  config3_rc = config3_cycle * std::ceil(double(2 * vn_number) / double(maeri->dn_bw));
  config3_ur = double(dnn_model->dnn_hidden->hidden_x * 2) / double(maeri->pe_size * config3_cycle);
  map->setcontrol_step(config_num, config3_cycle);
  map->setinput_trans(config_num, config3_cycle);
  map->setfilter_trans(config_num, config3_cycle);
  map->setcycle(config_num, config3_rc);
  map->setconfig_ur(config_num, config3_ur);
  CalculateDSN(map, config_num, 2);
  CalculateMSN(map, config_num,2);
  CalculateRSN(map, config_num);
  CalculateSPM(map, config_num, 2);
  CalculateEnergy(map, config_num);

  //Config 4, pure hardmard product
  config_num++;
  if(dnn_model->dnn_hidden->hidden_x > maeri->pe_size) {
    vn_number = maeri->pe_size;
  }
  else{
    vn_number = dnn_model->dnn_hidden->hidden_x;
  }
  peak_ur = double(vn_number) / (maeri->pe_size);
  map->setvn_num(config_num, vn_number);
  map->setelementinvn(config_num, 1);
  map->setinput_unicast(config_num, vn_number);
  map->setweight_unicast(config_num, vn_number);
  map->setweight_multicast(config_num, 0);
  map->setinput_multicast(config_num, 0);
  map->setweightmulticast_size(config_num, 0);
  map->setinputmulticast_size(config_num, 0);
  map->setpeak_ur(config_num, peak_ur);
  config4_inst = CalculateLSTMCt_Inst(2, 1);
  map->setcode_size(config_num, config4_inst);
  config4_cycle = std::ceil(double(dnn_model->dnn_hidden->hidden_x) / double(maeri->pe_size));
  config4_rc = config4_cycle * std::ceil(double(vn_number) / double(maeri->pe_size));
  map->setcontrol_step(config_num, config4_cycle);
  map->setinput_trans(config_num, config4_cycle);
  map->setfilter_trans(config_num, config4_cycle);
  map->setcycle(config_num, config4_rc);
  config4_ur = double(dnn_model->dnn_hidden->hidden_x) / double(maeri->pe_size * config4_cycle);
  map->setconfig_ur(config_num, config4_ur);
  CalculateDSN(map, config_num, 2);
  CalculateMSN(map, config_num,2);
  CalculateRSN(map, config_num);
  CalculateSPM(map, config_num, 2);
  CalculateEnergy(map, config_num);
  if(dnn_model->cnn_input->input_x + dnn_model->dnn_hidden->hidden_x <= maeri->pe_size) {
    total_ur = (config1_cycle * config1_ur + config3_cycle * config3_ur + config4_cycle * config4_ur)
             / double(config1_cycle + config3_cycle + config4_cycle);
  }
  else {
    total_ur = (config1_cycle * config1_ur + config2_cycle * config2_ur + config3_cycle * config3_ur + config4_cycle * config4_ur)
             / double(config1_cycle + config2_cycle + config3_cycle + config4_cycle);
  }
  map->setaverage_ur(total_ur);
  CalculateDram(map, partial, 0);
  map->setconfig_num(config_num+1);
  map->sum_all();
  mappings[map_num] = map;
  map_num++;
  setmapping_num(map_num);
  writeProfile(profile, mappings);
}

void Analyzer::writeCambrion(std::ofstream& profile, std::vector<std::pair<int, double> > info) {
  int i = 1;
  for(std::vector<std::pair<int, double> >::iterator I = info.begin(), E = info.end(); I != E; I++) {
    if(I == E - 1) {
      int cycle = I->first;
      double ur = I->second;
      profile << "Cambricon total cycle is : " << cycle << "\n";
      profile << "Cambricon total average ur is: " << ur << "\n";
    }
    else {
      int cycle = I->first;
      double ur = I->second;
      profile << "Cambricon " << i << "th phase cycle is: " << cycle << "\n";
      profile << "Cambricon " << i << "th phase ur is: " << ur << "\n";
    }
    i++;
  }
}

void Analyzer::AnalyzeCambricon(std::ofstream& profile) {

  int filterx = dnn_model->cnn_filter->filter_x;
  int filtery = dnn_model->cnn_filter->filter_x;
  int filterc = dnn_model->cnn_filter->filter_channel;
  int filtern = dnn_model->cnn_filter->filter_number;
  int inputx  = dnn_model->cnn_input->input_x;
  int inputy  = dnn_model->cnn_input->input_y;
  int inputc  = dnn_model->cnn_input->input_channel;
  int outputx = dnn_model->cnn_output->output_x;
  int outputy = dnn_model->cnn_output->output_y;
  int outputc = dnn_model->cnn_output->output_channel;

  int rowsize = 32;
  int colsize = 8;

  std::vector<std::pair<int, double> > info;
  int total_cycle = 0;
  double total_ur = 0;

  if(filterx * filtery * filterc < rowsize) {
    int filnum = std::ceil(double(filtern) / double(colsize));
    int part1_cycle = outputx * outputy * filnum;
    double part1_ur = double(filterx * filtery * filterc * outputx * outputy * outputc) / double(part1_cycle * maeri->pe_size);
    std::pair<int, double> part1info(part1_cycle, part1_ur);
    info.push_back(part1info);
    writeCambrion(profile, info);
  }
  else if(filterx * filtery < rowsize) {
    int kernelch = std::floor(double(rowsize) / double(filterx * filtery));
    int psnum = std::ceil(double(filterc) / double(kernelch));
    int filnum = std::ceil(double(filtern) / double(colsize));
    int part1_cycle = outputx * outputy * filnum * psnum;
    double mul = filterx * filtery * filterc * outputx * outputy * outputc;
    unsigned int totalmul = part1_cycle * maeri->pe_size;
    double part1_ur = mul / totalmul;

    std::pair<int, double> part1info(part1_cycle, part1_ur);
    info.push_back(part1info);

    int psnum2 = std::ceil(double(psnum) / double(rowsize));
    int part2_cycle = std::ceil(double(outputx * outputy * outputc) / double(colsize)) * psnum2;
    double part2_ur = double(outputx * outputy * outputc * psnum) / double(part2_cycle * maeri->pe_size);

    std::pair<int, double> part2info(part2_cycle, part2_ur);
    info.push_back(part2info);

    if(psnum > rowsize) {
      int part3_cycle = std::ceil(double(outputx * outputy * outputc) / double(colsize)) * std::ceil(double(psnum2) / double(rowsize));
      double part3_ur = double(outputx * outputy * outputc * psnum2) / double(part3_cycle * maeri->pe_size);

      total_cycle = part1_cycle + part2_cycle +part3_cycle;
      total_ur = double(part1_cycle * part1_ur + part2_cycle * part2_ur + part3_cycle * part3_ur) / double(part1_cycle + part2_cycle + part3_cycle);

      std::pair<int, double> part3info(part3_cycle, part3_ur);
      info.push_back(part3info);
      std::pair<int, double> totalinfo(total_cycle, total_ur);
      info.push_back(totalinfo);
      writeCambrion(profile, info);
      if(psnum2 > rowsize) {
        std::cerr << "Currently, we don't support channel size  larger than 1024 !\n";
      }
    }
    else {
      total_cycle = part1_cycle + part2_cycle;
      total_ur = double(part1_ur * part1_cycle + part2_ur * part2_cycle) / double(part1_cycle + part2_cycle);
      std::pair<int, double> totalinfo(total_cycle, total_ur);
      info.push_back(totalinfo);
      writeCambrion(profile, info);
    }
  }
  else if(filterx < rowsize) {
    int psnum = 0;
    int kernelch = std::floor(double(rowsize) / double(filterx));
    if(filtery % kernelch ==0 || filterc % kernelch == 0) {
      psnum = filterc * filtery / kernelch;
    }
    else if (filtery / kernelch > filterc / kernelch) {
      psnum = std::ceil(double(filtery) / double(kernelch)) * filterc;
    }
    else{
      psnum = std::ceil(double(filterc) / double(kernelch)) * filtery;
    }
    int part1_cycle = std::ceil(double(outputc) / double(colsize)) * outputx * outputy * psnum;
    double part1_ur = double(outputx * outputy * outputc * filterx * filtery *filterc) / double(part1_cycle * maeri->pe_size);
    std::pair<int, double> part1info(part1_cycle, part1_ur);
    info.push_back(part1info);

    int psnum2 = std::ceil(double(psnum) / double(rowsize));
    int part2_cycle = std::ceil(double(outputx * outputy * outputc) / double(colsize)) * psnum2;
    double part2_ur = double(outputx * outputy * outputc * psnum) / double(part2_cycle * maeri->pe_size);

    std::pair<int, double> part2info(part2_cycle, part2_ur);
    info.push_back(part2info);

    total_cycle = part1_cycle + part2_cycle;
    total_ur = double(part1_ur * part1_cycle + part2_ur * part2_cycle) / double(part1_cycle + part2_cycle);
    std::pair<int, double> totalinfo(total_cycle, total_ur);
    info.push_back(totalinfo);


    if(psnum > rowsize) {
      int part3_cycle = std::ceil(double(outputx * outputy * outputc) / double(colsize)) * std::ceil(double(psnum2) / double(rowsize));
      int part3_ur = double(outputx * outputy * outputc * psnum2) / double(part3_cycle * maeri->pe_size);

      total_cycle = part1_cycle + part2_cycle +part3_cycle;
      total_ur = double(part1_cycle * part1_ur + part2_cycle * part2_ur + part3_cycle * part3_ur) / double(part1_cycle + part2_cycle + part3_cycle);

      std::pair<int, double> part3info(part3_cycle, part3_ur);
      info.push_back(part3info);
      std::pair<int, double> totalinfo(total_cycle, total_ur);
      info.push_back(totalinfo);
      writeCambrion(profile, info);
      if(psnum2 > rowsize) {
        std::cerr << "Currently, we don't support channel size  larger than 1024 !\n";
      }
    }
    else {
      writeCambrion(profile, info);
    }
  }
}

void Analyzer::AnalyzeCambriconFC(std::ofstream& profile) {
  int filterx = dnn_model->cnn_filter->filter_x;
  int filtery = dnn_model->cnn_filter->filter_y;
  int filterc = dnn_model->cnn_filter->filter_channel;
  int filtern = dnn_model->cnn_filter->filter_number;
  int inputx  = dnn_model->cnn_input->input_x;
  int inputy  = dnn_model->cnn_input->input_y;
  int inputc  = dnn_model->cnn_input->input_channel;
  int outputx = dnn_model->cnn_output->output_x;
  int outputy = dnn_model->cnn_output->output_y;
  int outputc = dnn_model->cnn_output->output_channel;

  int rowsize = 32;
  int colsize = 8;

  int total_cycle = 0;
  double total_ur = 0;
  std::vector<std::pair<int, double> > info;

  int psnum = std::ceil(double(inputx) / (rowsize));
  int filnum = std::ceil(double(filterx) / double(colsize));
  int part1_cycle = psnum * filnum;
  double part1_ur = double(filterx * filtery) / double(part1_cycle * maeri->pe_size);
  std::pair<int, double> part1info(part1_cycle, part1_ur);
  info.push_back(part1info);
  if(inputx > rowsize) {
    int psnum1 = std::ceil(double(psnum) / double(rowsize));
    int part2_cycle = psnum1 * filnum;
    double part2_ur = double(psnum * filterx) / double(part2_cycle * maeri->pe_size);
    std::pair<int, double> part2info(part2_cycle, part2_ur);
    info.push_back(part2info);
    if(psnum > rowsize) {
      int psnum2 = std::ceil(double(psnum1) / double(rowsize));
      int part3_cycle = psnum2 * filnum;
      double part3_ur = double(psnum1 * filterx) / double(part3_cycle * maeri->pe_size);
      std::pair<int, double> part3info(part3_cycle, part3_ur);
      info.push_back(part3info);
      total_cycle = part1_cycle + part2_cycle + part3_cycle;
      total_ur = double(part1_cycle * part1_ur + part2_cycle * part2_ur + part3_cycle * part3_ur) / double(part1_cycle + part2_cycle + part3_cycle);
      std::pair<int, double> totalinfo(total_cycle, total_ur);
      info.push_back(totalinfo);
      writeCambrion(profile, info);
      if(psnum2 > rowsize) {
        std::cerr << "We don't support input size larger than 32768 !\n";
      }
    }
    else {
      total_cycle = part1_cycle + part2_cycle;
      total_ur = double(part1_cycle * part1_ur + part2_cycle * part2_ur) / double(part1_cycle + part2_cycle);
      std::pair<int, double> totalinfo(total_cycle, total_ur);
      info.push_back(totalinfo);
      writeCambrion(profile, info);
    }
  }
  else {
    writeCambrion(profile, info);
  }
}

void Analyzer::writeSystolic(std::ofstream& profile, std::map<std::pair<int, int>, double> arraylist) {
  for(std::map<std::pair<int, int>, double>::iterator I = arraylist.begin(), E = arraylist.end(); I != E; I++) {
    std::pair<int, int> array = I->first;
    double ur = I->second;
    profile << "Systolic array row: " << array.first << " PEs, column: " << array.second << " PEs.\n";
    profile << "Utilization rate under this array size is: " << ur << "\n\n";
  }
}

void Analyzer::AnalyzeSystolicCNN(std::ofstream &profile) {
  int outputnum = dnn_model->cnn_output->output_x * dnn_model->cnn_output->output_y;
  int filternum = dnn_model->cnn_filter->filter_number;

  std::vector<std::pair<int, int> > systolic;
  std::map<std::pair<int, int>, double> arrayur;
  int n = 0;
  for (n = 1; n*n <= maeri->pe_size; n++) {
    if(maeri->pe_size % n == 0) {
      int rownum = n;
      int colnum = maeri->pe_size / n;
      std::pair<int, int> rowandcol(rownum, colnum);
      systolic.push_back(rowandcol);
    }
  }

  for(std::vector<std::pair<int, int> >::iterator I = systolic.begin(), E = systolic.end(); I != E; I++) {
    std::pair<int, int> rowandcol = *I;
    int sysrow = I->first;
    int syscol = I->second;

    int bigrow1 = std::ceil(double(outputnum) / double(sysrow)) * sysrow;
    int bigcol1 = std::ceil(double(filternum) / double(syscol)) * syscol;
    double ur1 = double(outputnum * filternum) / double(bigrow1 * bigcol1);

    int bigrow2 = std::ceil(double(outputnum) / double(syscol)) * syscol;
    int bigcol2 = std::ceil(double(filternum) / double(sysrow)) * sysrow;
    double ur2 = double(outputnum * filternum) / double(bigrow2 * bigcol2);

    if (ur1 > ur2) {
      arrayur[rowandcol] = ur1;
    } else {
      arrayur[rowandcol] = ur2;
    }
  }
  writeSystolic(profile, arrayur);
}

void Analyzer::AnalyzeSystolicFC(std::ofstream &profile) {
  int outputnum = 1;
  int filternum = dnn_model->cnn_filter->filter_x;

  std::vector<std::pair<int, int> > systolic;
  std::map<std::pair<int, int>, double> arrayur;
  int n = 0;
  for (n = 1; n*n <= maeri->pe_size; n++) {
    if(maeri->pe_size % n == 0) {
      int rownum = n;
      int colnum = maeri->pe_size / n;
      std::pair<int, int> rowandcol(rownum, colnum);
      systolic.push_back(rowandcol);
    }
  }

  for(std::vector<std::pair<int, int> >::iterator I = systolic.begin(), E = systolic.end(); I != E; I++) {
    std::pair<int, int> rowandcol = *I;
    int colnum = I->second;
    int rownum = I->first;
    int bigcol = std::ceil(double(filternum)/ double(colnum)) * colnum;
    double ur = double(filternum) / double(bigcol * rownum);
    arrayur[rowandcol] = ur;
  }
  writeSystolic(profile, arrayur);
}

void Analyzer::ConfigGen(std::ofstream& config) {
  if(dnn_model->layer_type == "CONV") {
    config << "MAERI Configuration: \n";
    config << "Layer variables: \n";
    config << "X = " << dnn_model->cnn_input->input_x << "\n";
    config << "Y = " << dnn_model->cnn_input->input_y << "\n";
    config << "C = " << dnn_model->cnn_input->input_channel << "\n";
    config << "K = " << dnn_model->cnn_filter->filter_number << "\n";
    config << "N = " << dnn_model->cnn_input->input_batch << "\n";
    config << "X' = " << dnn_model->cnn_output->output_x << "\n";
    config << "Y' = " << dnn_model->cnn_output->output_y << "\n";
    config << "R = " << dnn_model->cnn_filter->filter_x << "\n";
    config << "S = " << dnn_model->cnn_filter->filter_y << "\n";

    if(bestmap) {
      config << "Mapping kernel (tile): \n";
      config << "T_X = " << bestmap->kernel_x << "\n";
      config << "T_Y = " << bestmap->kernel_y << "\n";
      config << "T_C = " << bestmap->kernel_c << "\n";
      config << "T_K = " << bestmap->kernel_n << "\n";
      config << "T_N = " << bestmap->kernel_in << "\n";
      config << "T_X' = " << bestmap->kernel_ox << "\n";
      config << "T_Y' = " << bestmap->kernel_oy << "\n";
    }

    config << "Virtual Neuron : \n";
    config << "VN_Size = " << bestmap->elementinvn[0] << "\n";
    config << "Num_VN = " << bestmap->vn_num[0] << "\n";

    config << "Loops outside the tile: \n";
    int outloop_x = dnn_model->cnn_filter->filter_x / bestmap->kernel_x;
    int outloop_y = dnn_model->cnn_filter->filter_y / bestmap->kernel_y;
    int outloop_c = dnn_model->cnn_input->input_channel / bestmap->kernel_c;
    int outloop_k = dnn_model->cnn_filter->filter_number / bestmap->kernel_n;
    int outloop_n = dnn_model->cnn_input->input_batch / bestmap->kernel_in;
    int outloop_ox = dnn_model->cnn_output->output_x / bestmap->kernel_ox;
    int outloop_oy = dnn_model->cnn_output->output_y / bestmap->kernel_oy;

    int mod_x = dnn_model->cnn_filter->filter_x % bestmap->kernel_x;
    int mod_y = dnn_model->cnn_filter->filter_y % bestmap->kernel_y;
    int mod_c = dnn_model->cnn_input->input_channel % bestmap->kernel_c;
    int mod_k = dnn_model->cnn_filter->filter_number % bestmap->kernel_n;
    int mod_n = dnn_model->cnn_input->input_batch % bestmap->kernel_in;
    int mod_ox = dnn_model->cnn_output->output_x % bestmap->kernel_ox;
    int mod_oy = dnn_model->cnn_output->output_y % bestmap->kernel_oy;

    config << "R/T_X = " << outloop_x << "\n";
    config << "S/T_Y = " << outloop_y << "\n";
    config << "C/T_C = " << outloop_c << "\n";
    config << "C/T_K = " << outloop_k << "\n";
    config << "C/T_N = " << outloop_n << "\n";
    config << "C/T_OX = " << outloop_ox << "\n";
    config << "C/T_OY = " << outloop_oy << "\n";

    config << "R%T_X = " << mod_x << "\n";
    config << "S%T_Y = " << mod_y << "\n";
    config << "C%T_C = " << mod_c << "\n";
    config << "K%T_K = " << mod_k << "\n";
    config << "N%T_N = " << mod_n << "\n";
    config << "X'%T_OX = " << mod_ox << "\n";
    config << "Y'%T_OY = " << mod_oy << "\n";

    config << "Outer loop order (from outermost to innermost): N->C->K->X'->Y' \n";

    int x = 0;
    int y = 0;
    int c = 0;
    int k = 0;
    int n = 0;
    int ox = 0;
    int oy = 0;
    if (mod_x > 0) { x = outloop_x + 1; } else { x = outloop_x; }
    if (mod_y > 0) { y = outloop_y + 1; } else { y = outloop_y; }
    if (mod_c > 0) { c = outloop_c + 1; } else { c = outloop_c; }
    if (mod_k > 0) { k = outloop_k + 1; } else { k = outloop_k; }
    if (mod_n > 0) { n = outloop_n + 1; } else { n = outloop_n; }
    if (mod_ox > 0) { ox = outloop_ox + 1; } else { ox = outloop_ox; }
    if (mod_oy > 0) { oy = outloop_oy + 1; } else { oy = outloop_oy; }
    unsigned long long total_iter = x * y * c * k * n * ox * oy;
    config << "Total_iteration = " << total_iter << "\n";
  }
  else if (dnn_model->layer_type == "MAXPOOL") {

  }
  else if (dnn_model->layer_type == "FC") {

  }
  else if (dnn_model->layer_type == "RNN") {

  }




}
