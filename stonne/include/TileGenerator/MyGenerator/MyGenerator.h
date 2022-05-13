#ifndef STONNE_TILEGENERATOR_MYGENERATOR_H
#define STONNE_TILEGENERATOR_MYGENERATOR_H

#include "TileGenerator/Utils/Target.h"
#include "TileGenerator/Utils/Tiles.h"

using namespace TileGenerator;

namespace MyGenerator {

    class MyGenerator {
    private:
        uint num_ms;
        uint dn_bw;
        uint rn_bw;

    public:
        MyGenerator(int num_ms, int dn_bw, int rn_bw) : num_ms(num_ms), dn_bw(dn_bw), rn_bw(rn_bw) {}
        ~MyGenerator() = default;

        DenseGemmTile generateDenseGemmTile(uint M, uint N, uint K, Target target);

        SparseDenseTile generateSparseDenseTile(uint M, uint N, uint K, float MK_sparsity);

    };

} // namespace MyGenerator

#endif //STONNE_TILEGENERATOR_MYGENERATOR_H
