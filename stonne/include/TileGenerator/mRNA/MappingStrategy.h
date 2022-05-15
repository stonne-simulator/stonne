//
// Created by Zhongyuan Zhao on 9/22/18.
//

#ifndef MAPPINGSTRATEGY_H_
#define MAPPINGSTRATEGY_H_

#include <map>
#include <vector>
#include <iostream>
#include "TileGenerator/mRNA/define.h"

namespace mRNA {

    class MappingStrategy {
    public:
        MappingStrategy() {
            kernel_x = 0;
            kernel_y = 0;
            kernel_c = 0;
            kernel_n = 0;
            average_ur = 0;
            config_num = 0;
            dram_access = 0;
            kernel_in = 1;
            kernel_ox = 1;
            kernel_oy = 1;
        }

        int kernel_x;
        int kernel_y;
        int kernel_c;
        int kernel_n;
        int kernel_in;
        int kernel_ox;
        int kernel_oy;

        //the attribute of each configuration
        std::map<int, ConfigType> config_attribute;
        //number of virtual neurons
        std::map<int, int> vn_num;
        //number of elements inside the virtual neurons
        std::map<int, int> elementinvn;

        std::map<int, int> code_size;

        //on chip space of the corresponding mapping strategy
        double onchip_input;
        double onchip_weight;
        double onchip_output;
        double onchip_space;
        //maximum utilization rate of each configuration
        std::map<int, double> peak_ur;
        //average utilization rate throughout the whole configuration
        std::map<int, double> config_ur;
        //average utilization rate throughout the whole layer.
        double average_ur;
        //the access count of the interconnect between distribute switches inside distribute network in each configuration
        std::map<int, unsigned long long> dn_access;
        //number of the unicast for input
        std::map<int, std::vector<int> > input_unicast;
        //number of the unicast for weight
        std::map<int, std::vector<int> > weight_unicast;
        //number of the multicast for input
        std::map<int, std::vector<int> > input_multicast;
        //number of the multicast for weight
        std::map<int, std::vector<int> > weight_multicast;
        //size of input multicast
        std::map<int, std::vector<int> > inputmulticast_size;
        //size of weight multicast
        std::map<int, std::vector<int> > weightmulticast_size;
        //the local register access count of multiplier network in each configuration
        std::map<int, unsigned long long> msreg_access;
        //the access count of the interconnect between multiplier switches inside multiplier network in each configuration
        std::map<int, unsigned long long> mslink_access;
        //the access count of the interconnect between reduce switches inside reduce network in each configuration
        std::map<int, unsigned long long> rn_access;
        //the count of the reduce operation in each configuration
        std::map<int, unsigned long long> reduce_num;
        //the count of the multiply operation in each configuration
        std::map<int, unsigned long long> multiply_num;
        //the count of scratchpad memory input read in each configuration
        std::map<int, unsigned long long> spminput_read;
        //the count of scratchpad memory weight read in each configuration
        std::map<int, unsigned long long> spmweight_read;
        //the count of scratchpad memory write in each configuration
        std::map<int, unsigned long long> spm_write;
        //the count of main memory access
        unsigned long long dram_access;

        double dram_eng;
        //number of configuration
        int config_num;
        //number of control steps in each configuration
        std::map<int, unsigned long long> config_cs;
        //number of cycles in each configuration
        std::map<int, unsigned long long> config_cycle;

        std::map<int, unsigned long long> input_trans;

        std::map<int, unsigned long long> filter_trans;

        std::map<int, double> spminputread_eng;

        std::map<int, double> spmweightread_eng;

        std::map<int, double> spmwrite_eng;

        std::map<int, double> msreg_eng;

        std::map<int, double> msaccess_eng;

        std::map<int, double> multiply_eng;

        std::map<int, double> reduce_eng;

        std::map<int, double> rsaccess_eng;

        std::map<int, double> dsaccess_eng;

        void setkernel_x(int x) { kernel_x = x; }

        void setkernel_y(int y) { kernel_y = y; }

        void setkernel_c(int c) { kernel_c = c; }

        void setkernel_n(int n) { kernel_n = n; }

        void setonchip_input(double input) { onchip_input = input; }

        void setonchip_weight(double weight) { onchip_weight = weight; }

        void setonchip_output(double output) { onchip_output = output; }

        void setonchip_space(double space) { onchip_space = space; }

        void setdram_access(unsigned long long dram) { dram_access = dram; }

        void setaverage_ur(double ur) { average_ur = ur; }

        void setvn_num(int i, int vnnum);

        void setelementinvn(int i, int element);

        void setinput_unicast(int i, int inputunicast);

        void setweight_unicast(int i, int weightunicast);

        void setinput_multicast(int i, int inputmulticast);

        void setweight_multicast(int i, int weightmulticast);

        void setinputmulticast_size(int i, int size);

        void setweightmulticast_size(int i, int size);

        void setpeak_ur(int i, double ur);

        void setconfig_ur(int i, double ur);

        void setconfig_attribute(int i, ConfigType attribute);

        //All the functions below needs to calculate the total number in sum_all() function
        void setcode_size(int i, int size);

        void setcontrol_step(int i, unsigned long long cs);

        void setcycle(int i, unsigned long long cycle);

        void setinput_trans(int i, unsigned long long trans);

        void setfilter_trans(int i, unsigned long long trans);

        void setconfig_num(int num) { config_num = num; }

        void setdn_access(int i, unsigned long long dn);

        void setmslink_access(int i, unsigned long long mslink);

        void setmsreg_access(int i, unsigned long long msreg);

        void setmultiply_num(int i, unsigned long long multiply);

        void setrn_access(int i, unsigned long long rn);

        void setreduce_num(int i, unsigned long long reduce);

        void setspminput_read(int i, unsigned long long read);

        void setspmweight_read(int i, unsigned long long read);

        void setspm_write(int i, unsigned long long write);

        void setspminputread_energy(int i, double spminputread);

        void setspmweightread_energy(int i, double spmweightread);

        void setspmwrite_energy(int i, double spmwrite);

        void setmsreg_energy(int i, double msreg);

        void setmslink_energy(int i, double mslink);

        void setmultiply_energy(int i, double multiply);

        void setrsaccess_energy(int i, double rsaccess);

        void setreduce_energy(int i, double reduce);

        void setdnaccess_energy(int i, double dnaccess);

        void setdram_energy(double dram) { dram_eng = dram; }

        unsigned long long gettotal_cycle();

        double gettotal_energy();

        double getenergy_efficiency();


        void sum_all();

    };

}

#endif //MAPPINGSTRATEGY_H_
