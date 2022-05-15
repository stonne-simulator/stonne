#ifndef STONNE_TILEGENERATOR_MRNAGENERATOR_H
#define STONNE_TILEGENERATOR_MRNAGENERATOR_H

#include "TileGenerator/Utils/Target.h"
#include "TileGenerator/Utils/Tiles.h"
#include "TileGenerator/mRNA/Analyzer.h"

using namespace TileGenerator;


namespace mRNA {

    /**
     * MrnaGenerator abstracts the generation of tiles using the mRNA tool.
     * At the moment, mRNA only supports the generation of CONV and DenseGEMM/FC layers.
     */
    class MrnaGenerator {
    public:
        /*******************************/
        /*** Tile Generation Methods ***/
        /*******************************/

        // CONV
        static ConvTile generateConvTile(uint num_ms, uint dn_bw, uint rn_bw,
                                         uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_,
                                         uint stride, Target target);

        // DenseGEMM/FC
        static DenseGemmTile generateDenseGemmTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, Target target);

    private:
        /*************************************/
        /*** Helper functions declarations ***/
        /*************************************/

        // mRNA model creation
        static DNNModel createDNNModel(std::string layer_type, uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y,
                                      uint X_, uint Y_, uint stride);
        static DNNModel createConvModel(uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y,
                                       uint X_, uint Y_, uint stride);
        static DNNModel createDenseGemmModel(uint M, uint N, uint K);

        // output filename generation
        static std::string getMrnaOutputFilename(uint num_ms, uint dn_bw, uint rn_bw, uint R, uint S, uint C, uint K, uint G, uint N,
                                          uint X, uint Y, uint X_, uint Y_, uint stride);
        static std::string getMrnaOutputFilename(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K);
    };

} // namespace mRNA

#endif //STONNE_TILEGENERATOR_MRNAGENERATOR_H
