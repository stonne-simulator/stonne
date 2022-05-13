#include <iostream>
#include <cassert>
#include "TileGenerator/TileGenerator.h"
#include "TileGenerator/mRNA/Analyzer.h"
#include "TileGenerator/MyGenerator/MyGenerator.h"

// #define USE_DENSEGEMM_MRNA

namespace TileGenerator {

    /*************************************/
    /*** Helper functions declarations ***/
    /*************************************/

    mRNA::DNNModel createDNNModel(std::string layer_type, uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y,
                                    uint X_, uint Y_, uint stride);

    mRNA::DNNModel createConvModel(uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y,
                                   uint X_, uint Y_, uint stride);

    mRNA::DNNModel createDenseGemmModel(uint M, uint N, uint K);



    /******************************/
    /*** Tile Generator Methods ***/
    /******************************/

    ConvTile TileGenerator::generateConvTile(uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y,
                                             uint X_, uint Y_, uint stride, Target target) {
        std::cout << "Using mRNA as Tile Generator for CONV layer" << std::endl;


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


        // ofstream for mRNA logs, but currenly it's not used, so no output is generated
        std::ofstream paper_bin;
        // std::ofstream mRNA_output("mRNA_output.txt"); // example redirecting the output to a file

        // *** Generates the best tile mapping for the layer type
        // Executes mRNA algorithm
        analyzer.AnalyzeCNN(paper_bin, mRNA::OptGoal(target));
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
            // TODO: could happen that cannot generate a tile configuration for some special case?
            std::cerr << "Could not generate a Tile Configuration automatically" << std::endl;
            std::cerr << "Please, check the input parameters" << std::endl;
            assert(false);
        }

    }

#ifdef USE_DENSEGEMM_MRNA
    DenseGemmTile TileGenerator::generateDenseGemmTile(uint M, uint N, uint K, Target target) {
        std::cout << "Using mRNA as Tile Generator for DenseGemm layer" << std::endl;

        if (target == Target::NONE) {
            std::cerr << "Not mRNA target specified, aborting" << std::endl;
            assert(false);
        }

        // Note: if K > ms_num^2 it will not work and it's not supported yet in mRNA. Maybe it could change in the future
        //assert(K <= num_ms);
        // Check if mRNA can generate a tile for the given parameters, printing warning messages if not
        if (M * N * K > num_ms)
            std::cerr << "WARNING: M * N * K > num_ms , mRNA might not be able to generate a correct FC tile" << std::endl;
        if (K > num_ms * num_ms)
            std::cerr << "WARNING: K > num_ms^2 , mRNA might not be able to generate a correct FC tile" << std::endl;

        // *** Create the DNN model and configure the input, filter and output variables
        mRNA::DNNModel dnnModel = createDenseGemmModel(M, N, K);
        // *** Create the MAERI architecture
        mRNA::Maeri maeri(num_ms, dn_bw, rn_bw);
        // *** Create the Analyzer to generate the configuration
        mRNA::Analyzer analyzer(&maeri, &dnnModel, mRNA::OptGoal(target));


        // ofstream for mRNA logs, but currenly it's not used, so no output is generated
        std::ofstream paper_bin;
        // std::ofstream mRNA_output("mRNA_output.txt"); // example redirecting the output to a file

        // *** Generates the best tile mapping for the layer type
        // Executes mRNA algorithm
        analyzer.AnalyzeFC(paper_bin, mRNA::OptGoal(target));
        // Recovers and returns the best tile mapping for the layer configuration
        if (analyzer.mappings[0]) {
            return DenseGemmTile(
                 // In some strange cases mRNA was returning 0's in some fields of the tile
                std::max(1, analyzer.mappings[0]->kernel_y),  // T_M
                std::max(1, analyzer.mappings[0]->kernel_in), // T_N
                std::max(1, analyzer.mappings[0]->kernel_x)   // T_K
            );
        } else {
            // TODO: could happen that cannot generate a tile configuration for some special case?
            std::cerr << "Could not generate a Tile Configuration automatically" << std::endl;
            std::cerr << "Please, check the input parameters" << std::endl;
            assert(false);
        }
    }
#else
    DenseGemmTile TileGenerator::generateDenseGemmTile(uint M, uint N, uint K, Target target) {
        std::cout << "Using MyGenerator as Tile Generator for DenseGemm layer" << std::endl;

        // *** Create the generator
        MyGenerator::MyGenerator tileGenerator(num_ms, dn_bw, rn_bw);

        // *** Generates the best tile mapping for the layer type
        DenseGemmTile tile = tileGenerator.generateDenseGemmTile(M, N, K, target);

        // TODO: write messages

        return tile;
    }
#endif

    SparseDenseTile TileGenerator::generateSparseDenseTile(uint M, uint N, uint K, float MK_sparsity) {
        std::cout << "Using MyGenerator as Tile Generator for SparseDense layer" << std::endl;

        uint T_N = 1;
        uint T_K = 1;

        // TODO: implement this method

        return SparseDenseTile(T_N, T_K);
    }



    /************************************/
    /*** Helper functions definitions ***/
    /************************************/

    mRNA::DNNModel createDNNModel(std::string layer_type, uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_, uint stride) {
        assert(layer_type == "CONV" || layer_type == "FC");

        // *** Create the DNN model and configure the input, filter and output variables
        mRNA::DNNModel dnnModel;
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

    mRNA::DNNModel createConvModel(uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_, uint stride) {
        return createDNNModel("CONV", R, S, C, K, G, N, X, Y, X_, Y_, stride);
    }

    mRNA::DNNModel createDenseGemmModel(uint M, uint N, uint K) {
        return createDNNModel("FC", M, K, 1, 1, 1, N, K, 1, M, 1, 1);
    }

} // TileGenerator