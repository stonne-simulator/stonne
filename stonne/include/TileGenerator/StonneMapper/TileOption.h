#ifndef STONNE_STONNEMAPPER_TILEOPTION_H
#define STONNE_STONNEMAPPER_TILEOPTION_H

#include "TileGenerator/Utils/Tiles.h"

namespace StonneMapper {

    // A structure to hold the information about a possible tile to be used
    class TileOption {
    private:
        // used to compare two heuristics values and to be able to give a little more importance
        // to the higher T_N and T_K values in case of a heuristic tie
        constexpr static float heuristicEpsilon = 0.025f;
        // maximum extra value for heuristic if the value of T_N is used at it max possible value
        constexpr static float factorHeuristicN = 0.1f;
        // maximum extra value for heuristic if the value of T_K is used at it max possible value
        constexpr static float factorHeuristicK = 0.05f;

    public:
        uint T_M, T_N, T_K;
        float msUtilization;
        float edgeUtilization;
        // heuristic value used to select the best tile
        float heuristic;

        TileOption(uint num_ms, uint M, uint N, uint K, uint T_M, uint T_N, uint T_K);

        bool operator>(const TileOption &val) const;

        static float getEdgeUtilization(uint X, uint T_X);

    };

} // namespace StonneMapper

#endif //STONNE_STONNEMAPPER_TILEOPTION_H
