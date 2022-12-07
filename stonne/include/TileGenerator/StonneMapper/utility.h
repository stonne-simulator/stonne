#ifndef STONNE_STONNEMAPPER_UTILITY_H
#define STONNE_STONNEMAPPER_UTILITY_H

#include <iostream>
#include <set>
#include "TileGenerator/StonneMapper/TileOptionDenseGemm.h"
#include "TileGenerator/StonneMapper/TileOptionSparseDense.h"

namespace StonneMapper {

    /************************/
    /*** Helper functions ***/
    /************************/

    // Calculates the utilization on the edge on a dimension for a given tile
    float getEdgeUtilization(uint X, uint T_X);

    // Gets all the divisors of a given number
    std::set<uint> getAllDivisors(uint n);

    // Helper functions to generate all the tile options possible
    void addTilesPowersOfTwo(std::set<TileOptionDenseGemm> &options, uint M, uint N, uint K, uint num_ms);
    void addTilesUsingDivisors(std::set<TileOptionDenseGemm> &options, uint M, uint N, uint K, uint num_ms,
                               float minimumMsUtilization, float minimumEdgeUtilization);

    // Output file manipulation
    std::string getStonneMapperOutputFilename(std::string layerType, uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K);

    void writeDenseGemmGeneratorResults(std::set<TileOptionDenseGemm> &options, TileOptionDenseGemm &bestTile,
                                        uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K);

    void writeSparseDenseGeneratorResults(std::set<TileOptionSparseDense> &options, TileOptionSparseDense &bestTile,
                                        uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, float sparsity);

} // namespace StonneMapper

#endif //STONNE_STONNEMAPPER_UTILITY_H
