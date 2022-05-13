#ifndef STONNE_TILEGENERATOR_TILES_H
#define STONNE_TILEGENERATOR_TILES_H

typedef unsigned int uint;

namespace TileGenerator {

    class ConvTile {
    public:
        uint T_R;
        uint T_S;
        uint T_C;
        uint T_K;
        uint T_G;
        uint T_N;
        uint T_X_;
        uint T_Y_;

        ConvTile(uint T_R, uint T_S, uint T_C, uint T_K, uint T_G, uint T_N, uint T_X_, uint T_Y_) :
            T_R(T_R), T_S(T_S), T_C(T_C), T_K(T_K), T_G(T_G), T_N(T_N), T_X_(T_X_), T_Y_(T_Y_) {}
        ~ConvTile() = default;
    };

    class DenseGemmTile {
    public:
        uint T_M;
        uint T_N;
        uint T_K;

        DenseGemmTile(uint T_M, uint T_N, uint T_K) :
            T_M(T_M), T_N(T_N), T_K(T_K) {}
        ~DenseGemmTile() = default;
    };

    class SparseDenseTile {
    public:
        uint T_N;
        uint T_K;

        SparseDenseTile(uint T_N, uint T_K) :
            T_N(T_N), T_K(T_K) {}
        ~SparseDenseTile() = default;
    };

} // TileGenerator

#endif //STONNE_TILEGENERATOR_TILES_H
