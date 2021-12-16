#include "mRNA_Generator.h"
#include <cassert>

// Constructor used for CONV layers
mRNA_Generator::mRNA_Generator(Layer_t layer_type, int _ms_num, int _dn_bw, int _rn_bw, int R, int S, int C, int K, int G,
                               int N, int X, int Y, int X_, int Y_, int stride, mRNA::OptGoal _opt_goal) {
    init(layer_type, _ms_num, _dn_bw, _rn_bw, R, S, C, K, G, N, X, Y, X_, Y_, stride, _opt_goal);
}

// Constructor used for FC layers
mRNA_Generator::mRNA_Generator(Layer_t layer_type, int _ms_num, int _dn_bw, int _rn_bw, int M, int N, int K, mRNA::OptGoal _opt_goal) {
    // Note: if K > ms_num it will not work and it's not supported yet in mRNA. Maybe it could change in the future
    assert(K <= _ms_num);

    // Compared with an Model_Parameter.txt example from mRNA/input, we have inverted R and S because in the original it did not work either
    // Example of FC layer in mRNA/Input/vggnet/Model_parameter_19,txt
    init(layer_type, _ms_num, _dn_bw, _rn_bw, M, K, 1, 1, 1, N, K, 1, M, 1, 1, _opt_goal);
}

// Initializes the adapter between STONNE and mRNA
void mRNA_Generator::init(Layer_t layer_type, int _ms_num, int _dn_bw, int _rn_bw, int R, int S, int C, int K, int G, int N,
                     int X, int Y, int X_, int Y_, int stride, mRNA::OptGoal _opt_goal) {
    opt_goal = _opt_goal;

    // Stonne actually only supports mRNA for CONV and FC layers
    // mRNA tile generation could be expanded to RNN too in the future
    assert(layer_type == CONV
           || layer_type == FC
    // || layer_type == RNN
    );

    // *** Create the DNN model and configure the input, filter and output variables
    dnnModel = new mRNA::DNNModel();
    // Model main configuration and layer type
    dnnModel->model_name = "STONNE"; // it's not used, but neccesary to work
    dnnModel->layer_type = layert_mapping[layer_type]; // translate to mRNA string
    dnnModel->layer_num = "1"; // it's not used, but neccesary to work
    // Input parameters
    dnnModel->cnn_input->input_batch = N;
    dnnModel->cnn_input->input_x = X;
    dnnModel->cnn_input->input_y = Y;
    dnnModel->cnn_input->input_channel = C;
    // Filter parameters
    dnnModel->cnn_filter->filter_x = R;
    dnnModel->cnn_filter->filter_y = S;
    dnnModel->cnn_filter->filter_number = K;
    dnnModel->cnn_filter->filter_channel = C;
    dnnModel->cnn_filter->window_stride = stride;
    // Output parameters
    dnnModel->cnn_output->output_batch = N;
    dnnModel->cnn_output->output_x = X_;
    dnnModel->cnn_output->output_y = Y_;
    dnnModel->cnn_output->output_channel = K;
    // Hidden parameters, not really used
    dnnModel->dnn_hidden->hidden_x = 0;
    dnnModel->dnn_hidden->hidden_y = 0;
    dnnModel->dnn_hidden->hidden_channel = 0;

    // *** Create the MAERI architecture
    maeri = new mRNA::Maeri(_ms_num, _dn_bw, _rn_bw);

    // *** Create the Analyzer to generate the configuration
    analyzer = new mRNA::Analyzer(maeri, dnnModel, opt_goal);
}

mRNA_Generator::~mRNA_Generator() {
    delete dnnModel;
    delete maeri;
    delete analyzer;
}

/**
 * Executes the mRNA algorithm and calculates the optimum tile configuration for the configuration used
 * @return Tile with the mRNA generated configuration. Empty Tile in case of failure
 */
Tile mRNA_Generator::generateTileConfig() {
    // ofstream for mRNA logs, but currenly it's not used, so no output is generated
    std::ofstream paper_bin;
    // std::ofstream mRNA_output("mRNA_output.txt"); // example redirecting the output to a file

    // *** Generates the best tile mapping for the layer type
    // TODO: review if T_G and folding parameters need to be changed
    if (analyzer->dnn_model->layer_type == layert_mapping[CONV]) { // CONV layer
        // Executes mRNA algorithm
        analyzer->AnalyzeCNN(paper_bin, opt_goal);
        // Recovers and returns the best tile mapping for the layer configuration
        if (analyzer->bestmap) {
            return Tile(
                    analyzer->bestmap->kernel_x,  // T_X
                    analyzer->bestmap->kernel_y,  // T_Y
                    analyzer->bestmap->kernel_c,  // T_C
                    analyzer->bestmap->kernel_n,  // T_K
                    1,                            // T_G (default: 1)
                    analyzer->bestmap->kernel_in, // T_N
                    analyzer->bestmap->kernel_ox, // T_X'
                    analyzer->bestmap->kernel_oy, // T_Y'
                    false                         // folding (default: no)
            );
        }
    } else if (analyzer->dnn_model->layer_type == layert_mapping[FC]) { // FC layer
        // Executes mRNA algorithm
        analyzer->AnalyzeFC(paper_bin, opt_goal);
        // Recovers and returns the best tile mapping for the layer configuration
        if (analyzer->mappings[0]) {
            return Tile(
                    analyzer->mappings[0]->kernel_y,  // T_M
                    analyzer->mappings[0]->kernel_in, // T_N
                    analyzer->mappings[0]->kernel_x,  // T_K
                    false                             // folding (default: no)
            );
        }
    }
    // } else if (analyzer->dnn_model->layer_type == layert_mapping[RNN]) { // RNN layer
    //     analyzer->AnalyzeRNN(paper_bin, opt_goal);
    // }

    // Empty and wrong tile for error handling
    std::cerr << "mRNA could NOT generate a tile configuration !" << std::endl;
    return Tile(0, 0, 0, 0, 0, 0, 0, 0, false);
}