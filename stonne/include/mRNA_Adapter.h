#ifndef __MRNA_ADAPTER_H
#define __MRNA_ADAPTER_H

#include "mRNA/Analyzer.h"
#include "Tile.h"
using namespace mRNA;

// TODO: this class need to be refactorized

class mRNA_Adapter {
public:
    mRNA_Adapter(int _ms_num, int _dn_bw, int _rn_bw, int R, int S, int C, int K, int G, int N, int X, int Y, int X_, int Y_, int stride, OptGoal _opt_goal)
        : ms_num(_ms_num), dn_bw(_dn_bw), rn_bw(_rn_bw), opt_goal(_opt_goal) {
        dnnModel = new DNNModel("stonne", "CONV", "1", new CNNInput(N, X, Y, C), new CNNFilter(R, S, K, C, stride), new CNNOutput(N, X_, Y_, K), new RNNHidden(0,0,0));
        maeri = new Maeri(_ms_num, _dn_bw, _rn_bw);
        analyzer = new Analyzer(maeri, dnnModel, opt_goal);
    }

    int ms_num;
    int dn_bw;
    int rn_bw;
    DNNModel *dnnModel;
    Maeri *maeri;
    Analyzer *analyzer;
    OptGoal opt_goal;

    Tile getTileConfig() {
        std::string Type[3] = {"CONV", "FC", "RNN"};
        std::ofstream Profile_result("/dev/null"); // TODO: we can't do this. if we do it, it could not run on Windows
        if(analyzer->dnn_model->layer_type == Type[0]) {
            analyzer->AnalyzeCNN(Profile_result, opt_goal);
        }
        else if (analyzer->dnn_model->layer_type == Type[1]) { // TODO: analyze if this two last options are necessary
            analyzer->AnalyzeFC(Profile_result, opt_goal);
        }
        else if (analyzer->dnn_model->layer_type == Type[2]) {
            analyzer->AnalyzeRNN(Profile_result, opt_goal);
        }

        return analyzer->getTileConfig();
    }
};


#endif //__MRNA_ADAPTER_H
