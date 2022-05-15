//
// Created by Zhongyuan Zhao on 9/14/18.
//

#ifndef RSNETWORK_H_
#define RSNETWORK_H_

#include "TileGenerator/mRNA/MSNetwork.h"
#include <vector>
#include <map>
#include <cmath>
#include <iostream>

namespace mRNA {

    class RSwitch;

    class MSwitch;

    class Forwarder {
    public:
        Forwarder(int number) {
            fordnum = number;
            input_rs = NULL;
            config_rs = NULL;
        }

        int fordnum;
        RSwitch *input_rs;
        RSwitch *config_rs;

        MSwitch *input_ms;
        MSwitch *config_ms;

        void setPhyInput(RSwitch *inputrs) { input_rs = inputrs; }

        void setPhyInput(MSwitch *inputms) { input_ms = inputms; }

        int getfdnum() { return fordnum; }
    };

    class MSwitch;

    class RSwitch {
    public:
        RSwitch(int level, int num) {
            rs_level = level;
            rs_num = num;

            up_output = NULL;
            augmt_output = NULL;
            augmt_input = NULL;
            config_up = NULL;
            config_auginput = NULL;

            forward_output = NULL;
            config_forwd = NULL;

            linms = NULL;
            rinms = NULL;
            config_lms = NULL;
            config_rms = NULL;
        }

        int rs_level;
        int rs_num;

        RSwitch *up_output;
        RSwitch *augmt_output;
        RSwitch *augmt_input;
        RSwitch *config_up;
        RSwitch *config_auginput;
        RSwitch *config_augoutput;

        RSwitch *l_input;
        RSwitch *r_input;
        RSwitch *config_linput;
        RSwitch *config_rinput;

        Forwarder *forward_output;
        Forwarder *config_forwd;

        MSwitch *linms;
        MSwitch *rinms;
        MSwitch *config_lms;
        MSwitch *config_rms;

        int getrslev() { return rs_level; }

        int getrsnum() { return rs_num; }

        void setPhyInput(RSwitch *linput, RSwitch *rinput) {
            l_input = linput;
            r_input = rinput;
        }

        void setPhyInput(MSwitch *linput, MSwitch *rinput) {
            linms = linput;
            rinms = rinput;
        }

        void setPhyOutput(RSwitch *up) { up_output = up; }

        RSwitch *getPhyOutput() { return up_output; }

        void setPhyFOutput(Forwarder *forwd) { forward_output = forwd; }

        void setPhyAugment(RSwitch *augment) {
            augmt_output = augment;
            augmt_input = augment;
        }

        RSwitch *getPhyAOutput() { return augmt_output; }

        Forwarder *getPhyFOutput() { return forward_output; }
    };

    class RSNetwork {
    public:
        RSNetwork(int rn_bandwidth, int pe_size);

        int maxlev;
        int pesize;
        int rn_bw;
        std::map<std::pair<int, int>, RSwitch *> rswitchtable;
        std::map<int, Forwarder *> forwardertable;

        void setPhysicalConnection(int levelnumber);

        int getmaxlev() { return maxlev; }

        int getpesize() { return pesize; }
    };

}

#endif //RSNETWORK_H_
