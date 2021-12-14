#ifndef DNNMODEL_H_
#define DNNMODEL_H_

#include <string>
#include "utility.h"

namespace mRNA {

    class CNNInput {
    public:
        CNNInput() {}

        CNNInput(int _input_batch, int _input_x, int _input_y, int _input_channel)
                : input_batch(_input_batch), input_x(_input_x), input_y(_input_y), input_channel(_input_channel) {}

        int input_batch;
        int input_x;
        int input_y;
        int input_channel;
    };

    class CNNFilter {
    public:
        CNNFilter() {}

        CNNFilter(int _filter_x, int _filter_y, int _filter_number, int _filter_channel, int _window_stride)
                : filter_x(_filter_x), filter_y(_filter_y), filter_number(_filter_number),
                  filter_channel(_filter_channel), window_stride(_window_stride) {}

        int filter_x;
        int filter_y;
        int filter_number;
        int filter_channel;
        int window_stride;
    };

    class CNNOutput {
    public:
        CNNOutput() {}

        CNNOutput(int _output_batch, int _output_x, int _output_y, int _output_channel)
                : output_batch(_output_batch), output_x(_output_x), output_y(_output_y),
                  output_channel(_output_channel) {}

        int output_batch;
        int output_x;
        int output_y;
        int output_channel;
    };

    class RNNHidden {
    public:
        RNNHidden() {}

        RNNHidden(int _hidden_x, int _hidden_y, int _hidden_channel)
                : hidden_x(_hidden_x), hidden_y(_hidden_y), hidden_channel(_hidden_channel) {}

        int hidden_x;
        int hidden_y;
        int hidden_channel;
    };

//It should be noticed that CNNInput can represent all the input of different DNNModel type. Including fully connected layer, cnn layer and rnn layer.
    class DNNModel {
    public:
        DNNModel() {
            cnn_input = new CNNInput();
            cnn_filter = new CNNFilter();
            cnn_output = new CNNOutput();
            dnn_hidden = new RNNHidden();
        };

        DNNModel(std::string _model_name, std::string _layer_type, std::string _layer_num, CNNInput *_cnn_input,
                 CNNFilter *_cnn_filter, CNNOutput *_cnn_output, RNNHidden *_dnn_hidden)
                : model_name(_model_name), layer_type(_layer_type), layer_num(_layer_num), cnn_input(_cnn_input),
                  cnn_filter(_cnn_filter), cnn_output(_cnn_output), dnn_hidden(_dnn_hidden) {}

        std::string model_name;
        std::string layer_type;
        std::string layer_num;
        CNNInput *cnn_input;
        CNNFilter *cnn_filter;
        CNNOutput *cnn_output;
        RNNHidden *dnn_hidden;

        void parseModelName(std::istringstream &instr);

        void parseLayerType(std::istringstream &instr);

        void parseLayerNumber(std::istringstream &instr);

        void parseInput(std::ifstream &infile);

        void parseWeight(std::ifstream &infile);

        void parseOutput(std::ifstream &infile);

        void parseHidden(std::ifstream &infile);

        void parsefile(std::ifstream &infile);
    };

}

#endif
