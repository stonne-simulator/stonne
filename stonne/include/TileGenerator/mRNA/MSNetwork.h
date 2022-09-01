//
// Created by Zhongyuan Zhao on 9/14/18.
//

#ifndef MSNETWORK_H_
#define MSNETWORK_H_

#include "TileGenerator/mRNA/define.h"
#include "TileGenerator/mRNA/DSNetwork.h"
#include "TileGenerator/mRNA/RSNetwork.h"

namespace mRNA {

    class RSwitch;

    class DSwitch;

    class Forwarder;

    class MSwitch {
    public:
        MSwitch(int number) {
            ms_number = number;
            vn_id = 0;
            f_output = NULL;
            f_input = NULL;
            config_i = NULL;
            config_o = NULL;
            up_rs = NULL;
            config_rs = NULL;
            lds = NULL;
            rds = NULL;
            config_lds = NULL;
            config_rds = NULL;
        }

        MSwitch *f_output;
        MSwitch *f_input;
        MSwitch *config_i;
        MSwitch *config_o;

        RSwitch *up_rs;
        RSwitch *config_rs;

        DSwitch *lds;
        DSwitch *rds;
        DSwitch *config_lds;
        DSwitch *config_rds;

        Forwarder *forwd;
        Forwarder *config_forwd;

        int vn_id;
        int ms_number;
        Opcode operation;

        void setOpcode(Opcode op) { operation = op; }

        void setPhyFInput(MSwitch *src) { f_input = src; }

        void setPhyFOutput(MSwitch *dst) { f_output = dst; }

        void setPhyInput(DSwitch *l_ds, DSwitch *r_ds) {
            lds = l_ds;
            rds = r_ds;
        }

        void setPhyOutput(RSwitch *rs) { up_rs = rs; }

        RSwitch *getPhyOutput() { return up_rs; }

        void setPhyFOutput(Forwarder *fwd) { forwd = fwd; }

        Forwarder *getPhyFOutput() { return forwd; }

        void setVNid(int id) { vn_id = id; }

        int getmsnum() { return ms_number; }

        MSwitch *getPhyMInput() { return f_input; }
    };

    class MSNetwork {
    public:
        MSNetwork(int pe_size);

        std::map<int, MSwitch *> mswitchtable;

        void setPhysicalConnection(int pe_size);
    };

}

#endif //MSNETWORK_H_
