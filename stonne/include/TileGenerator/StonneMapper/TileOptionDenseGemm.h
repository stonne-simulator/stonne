#ifndef STONNE_STONNEMAPPER_TILEOPTIONDENSEGEMM_H
#define STONNE_STONNEMAPPER_TILEOPTIONDENSEGEMM_H

#include "TileGenerator/Utils/Tiles.h"

namespace StonneMapper {

    // A structure to hold the information about a possible DenseGEMM/FC tile to be used
    class TileOptionDenseGemm {
    private:
        // Used to compare two heuristics values and to be able to give a little more importance
        // to the higher T_N and T_K values in case of a heuristic tie
        constexpr static float heuristicEpsilon = 0.025f;
        // Maximum extra value for heuristic if the value of T_N is used at it max possible value
        constexpr static float factorHeuristicN = 0.1f;
        // Maximum extra value for heuristic if the value of T_K is used at it max possible value
        constexpr static float factorHeuristicK = 0.05f;

    public:
        // Tile dimensions
        uint T_M, T_N, T_K;
        // Percentage of the multiplier switches that will be used
        float msUtilization;
        // Percentage of operations in each dimension that will be useful
        float edgeUtilization;
        // Heuristic value used to select the best tile
        float heuristic;

        // Constructor. Calculates all the parameters used for the heuristic
        TileOptionDenseGemm(uint num_ms, uint M, uint N, uint K, uint T_M, uint T_N, uint T_K);

        // Basic comparator based on the stored tile
        bool operator==(const TileOptionDenseGemm &val) const;

        // Comparators based on the heuristic value
        bool operator<(const TileOptionDenseGemm &val) const;
        bool operator>(const TileOptionDenseGemm &val) const;
    };

} // namespace StonneMapper

#endif //STONNE_STONNEMAPPER_TILEOPTIONDENSEGEMM_H
