#ifndef STONNE_STONNEMAPPER_TILEOPTIONSPARSEDENSE_H
#define STONNE_STONNEMAPPER_TILEOPTIONSPARSEDENSE_H

#include "../utils/Tiles.hpp"

namespace StonneMapper {

// A structure to hold the information about a possible SparseDense tile to be used
class TileOptionSparseDense {
 private:
  // Used to compare two heuristics values and to be able to give a little more importance
  // to the higher T_N or T_K values in case of a heuristic tie
  constexpr static float heuristicEpsilon = 0.025f;

 public:
  // Tile dimensions
  std::size_t T_N, T_K;
  // Percentage of the multiplier switches that will be used
  float msUtilization;
  // Percentage of operations in each dimension that will be useful
  float edgeUtilization;
  // heuristic value used to select the best tile
  float heuristic;

  // Constructor. Calculates all the parameters used for the heuristic
  TileOptionSparseDense(std::size_t num_ms, std::size_t M, std::size_t N, std::size_t K, std::size_t T_N, std::size_t T_K, float sparsity);

  // Basic comparator based on the stored tile
  bool operator==(const TileOptionSparseDense& val) const;

  // Comparators based on the heuristic value
  bool operator<(const TileOptionSparseDense& val) const;
  bool operator>(const TileOptionSparseDense& val) const;
};

}  // namespace StonneMapper

#endif  //STONNE_STONNEMAPPER_TILEOPTIONSPARSEDENSE_H
