#ifndef STONNE_STONNEMAPPER_UTILITY_H
#define STONNE_STONNEMAPPER_UTILITY_H

#include <iostream>
#include <set>
#include "TileOptionDenseGemm.hpp"
#include "TileOptionSparseDense.hpp"

namespace StonneMapper {

/************************/
/*** Helper functions ***/
/************************/

// Calculates the utilization on the edge on a dimension for a given tile
float getEdgeUtilization(std::size_t X, std::size_t T_X);

// Gets all the divisors of a given number
std::set<std::size_t> getAllDivisors(std::size_t n);

// Helper functions to generate all the tile options possible
void addTilesPowersOfTwo(std::set<TileOptionDenseGemm>& options, std::size_t M, std::size_t N, std::size_t K, std::size_t num_ms);
void addTilesUsingDivisors(std::set<TileOptionDenseGemm>& options, std::size_t M, std::size_t N, std::size_t K, std::size_t num_ms, float minimumMsUtilization,
                           float minimumEdgeUtilization);

// Output file manipulation
std::string getStonneMapperOutputFilename(std::string layerType, std::size_t num_ms, std::size_t dn_bw, std::size_t rn_bw, std::size_t M, std::size_t N,
                                          std::size_t K);

void writeDenseGemmGeneratorResults(std::set<TileOptionDenseGemm>& options, TileOptionDenseGemm& bestTile, std::size_t num_ms, std::size_t dn_bw,
                                    std::size_t rn_bw, std::size_t M, std::size_t N, std::size_t K);

void writeSparseDenseGeneratorResults(std::set<TileOptionSparseDense>& options, TileOptionSparseDense& bestTile, std::size_t num_ms, std::size_t dn_bw,
                                      std::size_t rn_bw, std::size_t M, std::size_t N, std::size_t K, float sparsity);

}  // namespace StonneMapper

#endif  //STONNE_STONNEMAPPER_UTILITY_H
