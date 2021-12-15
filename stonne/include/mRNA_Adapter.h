#ifndef __MRNA_ADAPTER_H
#define __MRNA_ADAPTER_H

#include "mRNA/Analyzer.h"

using namespace mRNA;

// TODO: this class need to be refactorized

class mRNA_Adapter {
public:
    mRNA_Adapter(int _ms_num, int _dn_bw, int _rn_bw, int R, int S, int C, int K, int G, int N, int X, int Y, int X_,
                 int Y_, int stride, OptGoal _opt_goal)
            : ms_num(_ms_num), dn_bw(_dn_bw), rn_bw(_rn_bw), opt_goal(_opt_goal) {

        dnnModel = new DNNModel();

        dnnModel->model_name = "stonne";
        dnnModel->layer_type = "CONV";
        dnnModel->layer_num = "1";

        dnnModel->cnn_input->input_batch = N;
        dnnModel->cnn_input->input_x = X;
        dnnModel->cnn_input->input_y = Y;
        dnnModel->cnn_input->input_channel = C;

        dnnModel->cnn_filter->filter_x = R;
        dnnModel->cnn_filter->filter_y = S;
        dnnModel->cnn_filter->filter_number = K;
        dnnModel->cnn_filter->filter_channel = C;
        dnnModel->cnn_filter->window_stride = stride;

        dnnModel->cnn_output->output_batch = N;
        dnnModel->cnn_output->output_x = X_;
        dnnModel->cnn_output->output_y = Y_;
        dnnModel->cnn_output->output_channel = K;

        dnnModel->dnn_hidden->hidden_x = 0;
        dnnModel->dnn_hidden->hidden_y = 0;
        dnnModel->dnn_hidden->hidden_channel = 0;

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

        if (analyzer->dnn_model->layer_type == Type[0]) {
            analyzer->AnalyzeCNN(Profile_result, opt_goal);
        } else if (analyzer->dnn_model->layer_type == Type[1]) { // TODO: analyze if this two last options are necessary
            analyzer->AnalyzeFC(Profile_result, opt_goal);
        } else if (analyzer->dnn_model->layer_type == Type[2]) {
            analyzer->AnalyzeRNN(Profile_result, opt_goal);
        }

        return Tile(
                analyzer->bestmap->kernel_x,  // T_X
                analyzer->bestmap->kernel_y,  // T_Y
                analyzer->bestmap->kernel_c,  // T_C
                analyzer->bestmap->kernel_n,  // T_K
                1,                  // T_G (default: 1)
                analyzer->bestmap->kernel_in, // T_N
                analyzer->bestmap->kernel_ox, // T_X'
                analyzer->bestmap->kernel_oy, // T_Y'
                false             // folding (default: no)
        );
    }
};

#endif //__MRNA_ADAPTER_H
