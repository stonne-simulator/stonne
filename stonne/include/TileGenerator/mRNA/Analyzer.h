#ifndef Analyzer_H_
#define Analyzer_H_

#include "TileGenerator/mRNA/define.h"
#include "TileGenerator/mRNA/DNNModel.h"
#include "TileGenerator/mRNA/MAERIModel.h"
#include "TileGenerator/mRNA/MappingStrategy.h"
#include "TileGenerator/mRNA/utility.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <string>
#include <map>

namespace mRNA {

    class EnergyPara {
    public:
        EnergyPara() {}

        double dram;
        double spm;
        double reg;
        double ds_access;
        double ms_access;
        double rs_access;
        double multiply;
        double reduce;

        void setdram(double dram_eng) { dram = dram_eng; }

        void setspm(double spm_eng) { spm = spm_eng; }

        void setRegister(double reg_eng) { reg = reg_eng; }

        void setds_access(double ds_eng) { ds_access = ds_eng; }

        void setms_access(double ms_eng) { ms_access = ms_eng; }

        void setrs_access(double rs_eng) { rs_access = rs_eng; }

        void setmultiply(double multi) { multiply = multi; }

        void setreduce(double redu) { reduce = redu; }
    };

    class Analyzer {
    public:
        Analyzer(Maeri *m, DNNModel *model, OptGoal goal) {
            maeri = m;
            dnn_model = model;
            optgoal = goal;
            energypara = new EnergyPara();
            mapping_num = 0;
            bestmap = NULL;
        }

        EnergyPara *energypara;
        OptGoal optgoal;
        Maeri *maeri;
        DNNModel *dnn_model;
        std::map<int, MappingStrategy *> mappings;
        bool show_energy;
        int mapping_num;
        MappingStrategy *bestmap;

        //Kernelpara contains each configuration of the four parameters: x, y, c, n.
        void setoptgoal(OptGoal goal) { optgoal = goal; }

        void setshowenergy(bool energy) { show_energy = energy; }

        std::map<int, std::vector<int> > kernelpara;

        void setmapping_num(int num) { mapping_num = num; }

//  int getmapping_num() { return mapping_num; }
        void parseEnergyPara(std::ifstream &infile);

        void parseconfig(std::ifstream &config);

        int CalculateCNNConfig1_Inst(int kernelchannel,
                                     int kernelfilters);   // Calculate the instruction size of the first configuration.
        int CalculateCNNPartial_Inst(int config2vnnum);   // Calculate the instruction size of the second configuration.
        int CNNConfig1largefil_Inst(int kernel_y, int kernelfilters);

        void CalculateConfig1onchip(MappingStrategy *map, std::vector<double> partial, int type);

        double CalculateConfig1_UR(int cycle);

        double CalculatePartialsum(int psnum);

        //The energy analysis is based on the method in eyeriss: DRAM 200, On-chip SM
        void CalculateDram(MappingStrategy *map, std::vector<double> partialsum, int type);

        //calculate the count of spm read, write and total in each configuration
        void CalculateSPM(MappingStrategy *map, int config_num, int type);

        //calculate the count of ms communication, ms register access and multiply operation
        void CalculateMSN(MappingStrategy *map, int config_num, int type);

        //calculate the count of ds communication, dn_access is updated.
        void CalculateDSN(MappingStrategy *map, int config_num, int type);

        //calculate the count of rs communication
        void CalculateRSN(MappingStrategy *map, int config_num);

        //Calculate the energy of each module.
        void CalculateEnergy(MappingStrategy *map, int config_num);

        bool checkCNNInput();

        bool checkFCInput();

        bool checkRNNInput();

// The AnalyzeCNN function generate the profile result of:
//1. Instruction length of each configuration.
//2. On-chip space
//3. Partial sum scale.
//4. Computational resource utilization rate.
        void AnalyzeCNN(std::ofstream &profile, OptGoal option);

        int CalculateFCConfig1_Inst(int vn_num);

        int CalculateFCConfig2_Inst(int vn_num);

        void AnalyzeFC(std::ofstream &profile, OptGoal option);

        int CalculateRNNConfig1_Inst(int vn_num);

        int CalculateRNNConfig2_Inst(int vn_num);

        int CalculateLSTMCt_Inst(int stage, int vn_num);

        void AnalyzeRNN(std::ofstream &profile, OptGoal option);

        void writeProfile(std::ofstream &profile, std::map<int, MappingStrategy *> mappings);

        void writeCambrion(std::ofstream &profile, std::vector<std::pair<int, double> > info);

        void AnalyzeCambricon(std::ofstream &profile);

        void AnalyzeCambriconFC(std::ofstream &profile);

        void writeSystolic(std::ofstream &profile, std::map<std::pair<int, int>, double> arraylist);

        void AnalyzeSystolicCNN(std::ofstream &profile);

        void AnalyzeSystolicFC(std::ofstream &profile);

        static bool compMappingcycle(MappingStrategy *&m1, MappingStrategy *&m2);

        static bool compMappingEnergy(MappingStrategy *m1, MappingStrategy *m2);

        static bool compMappingEE(MappingStrategy *m1, MappingStrategy *m2);

        //Sortmapping and return the best one according to the optimizing option
        MappingStrategy *SortMappingStrategy(OptGoal goal);

        //Output the configuration of MAERI.
        void ConfigGen(std::ofstream &config);
    };

}

#endif
