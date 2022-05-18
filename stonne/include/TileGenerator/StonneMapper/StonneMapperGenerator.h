#ifndef STONNE_TILEGENERATOR_STONNEMAPPER_H
#define STONNE_TILEGENERATOR_STONNEMAPPER_H

#include "TileGenerator/Utils/Target.h"
#include "TileGenerator/Utils/Tiles.h"

using namespace TileGenerator;

namespace StonneMapper {

    /**
     * StonneMapper is a simple tool to generate automatically tiles for DenseGEMM/FC and SparseDense layers.
     */
    class StonneMapperGenerator {
    public:
        /*******************************/
        /*** Tile Generation Methods ***/
        /*******************************/

        // DenseGEMM/FC
        static DenseGemmTile generateDenseGemmTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, Target target);

        // SparseDense
        static SparseDenseTile generateSparseDenseTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, float MK_sparsity, Target target);
    };

} // namespace StonneMapper

#endif //STONNE_TILEGENERATOR_STONNEMAPPER_H
