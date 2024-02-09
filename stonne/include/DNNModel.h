#ifndef DNNMODEL_H_
#define DNNMODEL_H_

#include <string>
#include "utility.h"


class CNNInput {
public:
  CNNInput() {}
  int input_batch;
  int input_x;
  int input_y;
  int input_channel;
};

class CNNFilter {
public:
  CNNFilter() {}
  int filter_x;
  int filter_y;
  int filter_number;
  int filter_channel;
  int window_stride;
};

class CNNOutput {
public:
  CNNOutput() {}
  int output_batch;
  int output_x;
  int output_y;
  int output_channel;
};

class RNNHidden {
public:
  RNNHidden() {}
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
  std::string model_name;
  std::string layer_type;
  std::string layer_num;
  CNNInput* cnn_input;
  CNNFilter* cnn_filter;
  CNNOutput* cnn_output;
  RNNHidden* dnn_hidden;

  void parseModelName(std::istringstream& instr);
  void parseLayerType(std::istringstream& instr);
  void parseLayerNumber(std::istringstream& instr);
  void parseInput(std::ifstream& infile);
  void parseWeight(std::ifstream& infile);
  void parseOutput(std::ifstream& infile);
  void parseHidden(std::ifstream& infile);
  void parsefile(std::ifstream& infile);
};

#endif
