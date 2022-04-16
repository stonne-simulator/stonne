#ifndef __MRNA_ADAPTER_H
#define __MRNA_ADAPTER_H

#include "TileGenerator/mRNA/Analyzer.h"
#include "types.h"
#include "Tile.h"

/**
 * Class to interconnect STONNE with mRNA, acting as an intermediary between simulator and module
 */
class mRNA_Generator {
private:
    mRNA::DNNModel *dnnModel;
    mRNA::Maeri *maeri;
    mRNA::Analyzer *analyzer;
    mRNA::OptGoal opt_goal;

    // Helper struct to map from Stonne::Layer_t (enum) to mRNA::Layer_type (string)
    std::map<Layer_t, std::string> layert_mapping {
            {CONV, "CONV"},
            {FC,   "FC"}
            //{RNN,  "RNN"}
    };

    // For code reusing in constructors
    void init(Layer_t layer_type, int ms_num, int dn_bw, int rn_bw, int R, int S, int C, int K, int G, int N,
                   int X, int Y, int X_, int Y_, int stride, mRNA::OptGoal opt_goal);

public:
    // Constructor used for CONV layers
    mRNA_Generator(Layer_t layer_type, int ms_num, int dn_bw, int rn_bw, int R, int S, int C, int K, int G, int N,
                   int X, int Y, int X_, int Y_, int stride, mRNA::OptGoal opt_goal);

    // Constructor used for FC layers
    mRNA_Generator(Layer_t layer_type, int _ms_num, int _dn_bw, int _rn_bw, int M, int N, int K, mRNA::OptGoal opt_goal);

    ~mRNA_Generator();

    Tile generateTileConfig();
};

#endif //__MRNA_ADAPTER_H
