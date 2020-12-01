
#include "DNNModel.h"

void DNNModel::parseModelName(std::istringstream& instr) {
  std::string str;
  while(instr >> str) {
    if(isNum(str)){
      std::cerr << "Syntax error. Check Model_Name setting.";
    }
    else if(str == "=" || str == ":") {
      continue;
    }
    else {
      model_name = str;
    }
  }
}

void DNNModel::parseLayerType(std::istringstream& instr) {
  std::string str;
  while(instr >> str) {
    if(isNum(str)){
      std::cerr << "Syntax error. Check Layer Type setting.\n";
    }
    else if(str == "=" || str == ":") {
      continue;
    }
    else {
      layer_type = str;
    }
  }
}

void DNNModel::parseLayerNumber(std::istringstream& instr) {
  std::string str;
  while(instr >> str) {
    if(str == "=" || str == ":") {
      continue;
    }
    else {
      layer_num = str;
    }
  }

  if(layer_num == "") {
    std::cerr << "Can't get layer number. Check Layer Number setting.\n";
  }
}

void DNNModel::parseInput(std::ifstream& infile) {
  std::string buffer;
  while(getline(infile, buffer)) {
    if(buffer == "}") {
      break;
    }
    std::istringstream record(buffer);
    std::string str;
    while(record >> str) {
      if(isNum(str)) {
        std::cerr << "There should be variables to be set.\n";
      }
      else {
        if (str == "input_x") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_input->input_x = num;
          break;
        }
        else if (str == "input_y") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_input->input_y = num;
          break;

        }
        else if (str == "input_channel") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_input->input_channel = num;
          break;
        }
        else if(str == "input_batch") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_input->input_batch = num;
        }
        else {
          std::cerr << "Unsupported parameter keyword, please check syntax. ";
          break;
        }
      }
    }
  }
}

void DNNModel::parseWeight(std::ifstream& infile) {
  std::string buffer;
  while(getline(infile, buffer)) {
    if(buffer == "}"){
      break;
    }
    std::istringstream record(buffer);
    std::string str;
    while(record >> str) {
      if(isNum(str)) {
        std::cerr << "There should be variables to be set.\n";
      }
      else {
        if (str == "weight_x") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_filter->filter_x = num;
          break;
        }
        else if(str == "weight_y") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_filter->filter_y = num;
          break;
        }
        else if(str == "weight_channel") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_filter->filter_channel = num;
          break;
        }
        else if(str == "weight_number") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_filter->filter_number = num;
          break;
        }
        else if(str == "weight_stride") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_filter->window_stride = num;
          break;
        }
        else {
          std::cerr << "Unsupported parameter keyword, please check syntax. ";
          break;
        }
      }
    }
  }
}

void DNNModel::parseOutput(std::ifstream& infile) {
  std::string buffer;
  while(getline(infile, buffer)) {
    if(buffer == "}"){
      break;
    }
    std::istringstream record(buffer);
    std::string str;
    while(record >> str) {
      if(isNum(str)) {
        std::cerr << "There should be variables to be set.\n";
      }
      else {
        if (str == "output_x") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_output->output_x = num;
          break;
        }
        else if (str == "output_y") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_output->output_y = num;
          break;

        }
        else if (str == "output_channel") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_output->output_channel = num;
          break;
        }
        else if (str == "output_batch") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          cnn_output->output_batch = num;
          break;
        }
        else {
          std::cerr << "Unsupported parameter keyword, please check syntax. ";
          break;
        }
      }
    }
  }
}

void DNNModel::parseHidden(std::ifstream& infile) {
  std::string buffer;
  while(getline(infile, buffer)) {
    if(buffer == "}"){
      break;
    }
    std::istringstream record(buffer);
    std::string str;
    while(record >> str) {
      if(isNum(str)) {
        std::cerr << "There should be variables to be set.\n";
      }
      else {
        if (str == "hidden_x") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          dnn_hidden->hidden_x = num;
          break;
        }
        else if (str == "hidden_y") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          dnn_hidden->hidden_y = num;
          break;

        }
        else if (str == "hidden_channel") {
          std::string strnum = getstr(record);
          int num = std::stoi(strnum, nullptr, 10);
          dnn_hidden->hidden_channel = num;
          break;
        }
        else if(str == "}"){
          break;
        }
      }
    }
  }
}

void DNNModel::parsefile(std::ifstream& infile) {
  std::string buffer;
  while(getline(infile, buffer)) {
    std::istringstream record(buffer);
    std::string str;
    while(record >> str) {
      if(isNum(str)) {
        std::cerr << "Syntax error. Check digital position.\n";
      }
      else if(str == "Model_Name") {
        parseModelName(record);
      }
      else if(str == "Layer_Type") {
        parseLayerType(record);
      }
      else if(str == "Layer_Number") {
        parseLayerNumber(record);
      }
      else if(str == "Input_parameter") {
        parseInput(infile);
      }
      else if(str == "Weight_parameter") {
        parseWeight(infile);
      }
      else if(str == "Output_parameter") {
        parseOutput(infile);
      }
      else if(str == "Hidden_parameter") {
        parseHidden(infile);
      }
      else {
        continue;
      }
    }
  }
}
