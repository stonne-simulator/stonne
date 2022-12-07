//
// Created by Zhongyuan Zhao on 9/14/18.
//

#ifndef DSNETWORK_H_
#define DSNETWORK_H_

#include "TileGenerator/mRNA/MSNetwork.h"
#include <iostream>
#include <fstream>
#include <map>

namespace mRNA {

    class MSwitch;

    class DSwitch {
    public:
        DSwitch(int level_l, int num) {
            ds_level = level_l;
            ds_number = num;
            input_lds = NULL;
            input_rds = NULL;
            output_lds = NULL;
            output_rds = NULL;
            config_input = NULL;
            config_lds = NULL;
            config_rds = NULL;
            lms = NULL;
            rms = NULL;
            config_lms = NULL;
            config_rms = NULL;
        }

//Physical connection
        DSwitch *input_lds;
        DSwitch *input_rds;
        DSwitch *output_lds;
        DSwitch *output_rds;
//Software configuration
        DSwitch *config_input;
        DSwitch *config_lds;
        DSwitch *config_rds;

//The highest level layer of the distribute switches connect to multiplier switch.
        MSwitch *lms;
        MSwitch *rms;
        MSwitch *config_lms;
        MSwitch *config_rms;
//The data is used to test the function correctness of the distribute network.
        int data;
        int vn_id;

        int ds_level;
        int ds_number;
        int input_mode;
        int output_mode;

        void readdata(int d) { data = d; }

        int getdata() { return data; }

        void setvnid(int id) { vn_id = id; }

        int getvnid() { return vn_id; }

        void setInputMode(int imode) { input_mode = imode; }

        void setOutputMode(int omode) { output_mode = omode; }

        void setPhyLInput(DSwitch *l_input) { input_lds = l_input; }

        void setPhyRInput(DSwitch *r_input) { input_rds = r_input; }

        void setPhyOutput(DSwitch *left, DSwitch *right) {
            output_lds = left;
            output_rds = right;
        }

        void setPhyOutput(MSwitch *left, MSwitch *right) {
            lms = left;
            rms = right;
        }

        DSwitch *getPhyLOutput() { return output_lds; }

        DSwitch *getPhyROutput() { return output_rds; }

        MSwitch *getPhyLMS() { return lms; }

        MSwitch *getPhyRMS() { return rms; }

        int getAvOutput();

        void ConfigDS(int input_src, int output_mode);

        int getInputMode() { return input_mode; }

        int getOutputMode() { return output_mode; }
    };

    class DSNetwork {
    public:
        DSNetwork(double bw, int pe_size);

        int bandwidth;
        int rootSWsize;
        int maxlev;

        int getmaxlev() { return maxlev; }

        int getpesize() { return rootSWsize; }

        std::map<std::pair<int, int>, DSwitch *> dswitchtable;

        void setPhysicalConnection(int levelnum, int pe_size);

        void rootSWsread(int *array, int length, int interval);
    };

}

#endif //DSNETWORK_H_
