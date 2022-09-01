#ifndef STONNE_TILEGENERATOR_H
#define STONNE_TILEGENERATOR_H

#include "TileGenerator/Utils/Target.h"
#include "TileGenerator/Utils/Tiles.h"

namespace TileGenerator {

    /**
     * Currently supported generators (by default: CHOOSE_AUTOMATICALLY)
     */
    enum Generator {
        CHOOSE_AUTOMATICALLY = 0,
        MRNA = 1,
        STONNE_MAPPER = 2
    };

    /**
     * TileGenerator is the main class of the TileGenerator module.
     * It abstracts and unifies the tile generation process, hiding the tool used below.
     *
     * By default, it chooses automatically the best tool for each type of layer (Generator::CHOOSE_AUTOMATICALLY).
     * However, you can also indicate the specific tool you want to use indicating a Generator option.
     * Note that all tools are not designed to generate all types of mappings, so check out the tools available for each
     * type of layer before using this option.
     */
    class TileGenerator {
    private:
        uint num_ms;
        uint dn_bw;
        uint rn_bw;
        Generator generator;

    public:
        // Constructor and destructor
        TileGenerator(int num_ms, int dn_bw, int rn_bw, Generator generator = CHOOSE_AUTOMATICALLY)
                        : num_ms(num_ms), dn_bw(dn_bw), rn_bw(rn_bw), generator(generator) {}
        ~TileGenerator() = default;

        /*******************************/
        /*** Tile Generation Methods ***/
        /*******************************/

        // CONV
        ConvTile generateConvTile(uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_,
                                  uint stride, Target target);

        // DenseGEMM/FC
        DenseGemmTile generateDenseGemmTile(uint M, uint N, uint K, Target target);

        // SparseDense
        SparseDenseTile generateSparseDenseTile(uint M, uint N, uint K, float MK_sparsity, Target target);
    };

} // TileGenerator

#endif //STONNE_TILEGENERATOR_H