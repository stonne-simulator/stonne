#include <iostream>
#include <cassert>
#include "TileGenerator/mRNA/MrnaGenerator.h"


namespace mRNA {

    /*****************/
    /*** Constants ***/
    /*****************/
    const std::string mrnaParametersFilename = "parameters/mRNA_energy_parameters.txt";
    const std::string mrnaOutputBasename = "mRNA_output_";


    /******************************/
    /*** Tile Generator Methods ***/
    /******************************/

    ConvTile MrnaGenerator::generateConvTile(uint num_ms, uint dn_bw, uint rn_bw, uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_,
                                             uint stride, Target target) {

        if (target == Target::NONE) {
            std::cerr << "Not mRNA target specified, aborting" << std::endl;
            assert(false);
        }

        // Check if mRNA can generate a tile for the given parameters, printing warning messages if not
        if (R * S > num_ms)
            std::cerr << "WARNING: R * S > num_ms , mRNA might not be able to generate a CONV tile" << std::endl;
        if (R * C > num_ms)
            std::cerr << "WARNING: R * C > num_ms , mRNA might not be able to generate a CONV tile" << std::endl;


        // *** Create the DNN model and configure the input, filter and output variables
        mRNA::DNNModel dnnModel = createConvModel(R, S, C, K, G, N, X, Y, X_, Y_, stride);
        // *** Create the MAERI architecture
        mRNA::Maeri maeri(num_ms, dn_bw, rn_bw);
        // *** Create the Analyzer to generate the configuration
        mRNA::Analyzer analyzer(&maeri, &dnnModel, mRNA::OptGoal(target));
        // Loads the mRNA energy parameters into the analyzer if available
        std::ifstream configFile(mrnaParametersFilename);
        if (configFile.is_open()) {
            analyzer.parseconfig(configFile);
            configFile.close();
        } else {
            std::cerr << "WARNING: Could not open file " << mrnaParametersFilename <<
                      " with mRNA parameters, the tile that will be generated might not be as optimal as expected" << std::endl;
        }

        // ofstream for mRNA logs, saved in file
        std::ofstream mRNA_output(getMrnaOutputFilename(num_ms, dn_bw, rn_bw, R, S, C, K, G, N, X, Y, X_, Y_, stride));


        // *** Generates the best tile mapping for the layer type
        // Executes mRNA algorithm
        analyzer.AnalyzeCNN(mRNA_output, mRNA::OptGoal(target));
        // Recovers and returns the best tile mapping for the layer configuration
        if (analyzer.bestmap) {
            return ConvTile(
                    analyzer.bestmap->kernel_x,  // T_X
                    analyzer.bestmap->kernel_y,  // T_Y
                    analyzer.bestmap->kernel_c,  // T_C
                    analyzer.bestmap->kernel_n,  // T_K
                    1,                           // T_G (default: 1)
                    analyzer.bestmap->kernel_in, // T_N
                    analyzer.bestmap->kernel_ox, // T_X'
                    analyzer.bestmap->kernel_oy  // T_Y'
            );
        } else {
            std::cerr << "mRNA could not generate a Tile Configuration automatically" << std::endl;
            std::cerr << "Please, check the input parameters" << std::endl;
            assert(false);
        }

    }


    DenseGemmTile MrnaGenerator::generateDenseGemmTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, Target target) {
        if (target == Target::NONE) {
            std::cerr << "Not mRNA target specified, aborting" << std::endl;
            assert(false);
        }

        // Check if mRNA can generate a tile for the given parameters, printing warning messages if not
        // Note: if K > ms_num^2 it will not work and it's not supported yet in mRNA. Maybe it could change in the future
        if (K > num_ms * num_ms)
            std::cerr << "WARNING: K > num_ms^2 , mRNA might not be able to generate a correct FC tile" << std::endl;

        // *** Create the DNN model and configure the input, filter and output variables
        mRNA::DNNModel dnnModel = createDenseGemmModel(M, N, K);
        // *** Create the MAERI architecture
        mRNA::Maeri maeri(num_ms, dn_bw, rn_bw);
        // *** Create the Analyzer to generate the configuration
        mRNA::Analyzer analyzer(&maeri, &dnnModel, mRNA::OptGoal(target));
        // Loads the mRNA energy parameters into the analyzer if available
        std::ifstream configFile(mrnaParametersFilename);
        if (configFile.is_open()) {
            analyzer.parseconfig(configFile);
            configFile.close();
        } else {
            std::cerr << "WARNING: Could not open file " << mrnaParametersFilename <<
                      " with mRNA parameters, the tile that will be generated might not be as optimal as expected" << std::endl;
        }


        // ofstream for mRNA logs, saved in file
        std::ofstream mRNA_output(getMrnaOutputFilename(num_ms, dn_bw, rn_bw, M, N, K));

        // *** Generates the best tile mapping for the layer type
        // Executes mRNA algorithm
        analyzer.AnalyzeFC(mRNA_output, mRNA::OptGoal(target));
        // Recovers and returns the best tile mapping for the layer configuration
        if (analyzer.mappings[0]) {
            return DenseGemmTile(
                    // In some strange cases mRNA was returning 0's in some fields of the tile
                    std::max(1, analyzer.mappings[0]->kernel_y),  // T_M
                    std::max(1, analyzer.mappings[0]->kernel_in), // T_N (always: 1)
                    std::max(1, analyzer.mappings[0]->kernel_x)   // T_K
            );
        } else {
            std::cerr << "mRNA could not generate a Tile Configuration automatically" << std::endl;
            std::cerr << "Please, check the input parameters" << std::endl;
            assert(false);
        }
    }


    /************************************/
    /*** Helper functions definitions ***/
    /************************************/

    // Creates a mRNA DNNModel following the type configuration
    DNNModel MrnaGenerator::createDNNModel(std::string layer_type, uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_, uint stride) {
        assert(layer_type == "CONV" || layer_type == "FC");

        // *** Create the DNN model and configure the input, filter and output variables
        DNNModel dnnModel;
        // Model main configuration and layer type
        dnnModel.model_name = "STONNE"; // it's not used, but neccesary to work
        dnnModel.layer_type = layer_type; // translate to mRNA string
        dnnModel.layer_num = "1"; // it's not used, but neccesary to work
        // Input parameters
        dnnModel.cnn_input->input_batch = N;
        dnnModel.cnn_input->input_x = X;
        dnnModel.cnn_input->input_y = Y;
        dnnModel.cnn_input->input_channel = C;
        // Filter parameters
        dnnModel.cnn_filter->filter_x = R;
        dnnModel.cnn_filter->filter_y = S;
        dnnModel.cnn_filter->filter_number = K;
        dnnModel.cnn_filter->filter_channel = C;
        dnnModel.cnn_filter->window_stride = stride;
        // Output parameters
        dnnModel.cnn_output->output_batch = N;
        dnnModel.cnn_output->output_x = X_;
        dnnModel.cnn_output->output_y = Y_;
        dnnModel.cnn_output->output_channel = K;
        // Hidden parameters, not really used
        dnnModel.dnn_hidden->hidden_x = 0;
        dnnModel.dnn_hidden->hidden_y = 0;
        dnnModel.dnn_hidden->hidden_channel = 0;

        return dnnModel;
    }

    // Helper function to create DNNModel for CONV layer
    DNNModel MrnaGenerator::createConvModel(uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_, uint stride) {
        return createDNNModel("CONV", R, S, C, K, G, N, X, Y, X_, Y_, stride);
    }

    // Helper function to create DNNModel for DenseGEMM/FC layer
    DNNModel MrnaGenerator::createDenseGemmModel(uint M, uint N, uint K) {
        return createDNNModel("FC", M, K, 1, 1, 1, N, K, 1, M, 1, 1);
    }

    // Generates a filename for the results generation of a CONV layer
    std::string MrnaGenerator::getMrnaOutputFilename(uint num_ms, uint dn_bw, uint rn_bw,
                                      uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_, uint stride) {
        std::stringstream ss;
        ss << mrnaOutputBasename << "CONV" << "_num_ms" << num_ms << "_dn_bw" << dn_bw << "_rn_bw" << rn_bw <<
           "_R" << R << "_S" << S << "_C" << C << "_K" << K << "_G" << G << "_N" << N << "_X" << X << "_Y" << Y <<
           "_X_" << X_ << "_Y_" << Y_ << "_stride" << stride << ".txt";
        return ss.str();
    }

    // Generates a filename for the results generation of a DenseGEMM/FC layer
    std::string MrnaGenerator::getMrnaOutputFilename(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K) {
        std::stringstream ss;
        ss << mrnaOutputBasename << "FC" << "_num_ms" << num_ms << "_dn_bw" << dn_bw << "_rn_bw" << rn_bw <<
           "_M" << M << "_N" << N << "_K" << K << ".txt";
        return ss.str();
    }

} // namespace mRNA