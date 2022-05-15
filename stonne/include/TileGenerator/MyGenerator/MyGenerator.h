#ifndef STONNE_TILEGENERATOR_MYGENERATOR_H
#define STONNE_TILEGENERATOR_MYGENERATOR_H

#include "TileGenerator/Utils/Target.h"
#include "TileGenerator/Utils/Tiles.h"

using namespace TileGenerator;

namespace MyGenerator {

    /**
     * MyGenerator is a simple tool to generate automatically tiles for DenseGEMM/FC and SparseDense layers.
     */
    class MyGenerator {
    public:
        /*******************************/
        /*** Tile Generation Methods ***/
        /*******************************/

        // DenseGEMM/FC
        static DenseGemmTile generateDenseGemmTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, Target target);

        // SparseDense
        static SparseDenseTile generateSparseDenseTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, float MK_sparsity, Target target);

    private:
        /************************/
        /*** Helper functions ***/
        /************************/

        static float getEdgeUtilization(uint X, uint T_X);

        static void maximizeEdgeUtilization(uint M, uint N, uint K, uint &T_M, uint &T_N, uint &T_K);
    };

} // namespace MyGenerator

#endif //STONNE_TILEGENERATOR_MYGENERATOR_H
