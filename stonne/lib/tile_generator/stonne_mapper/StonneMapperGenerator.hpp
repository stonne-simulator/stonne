#ifndef STONNE_TILEGENERATOR_STONNEMAPPER_H
#define STONNE_TILEGENERATOR_STONNEMAPPER_H

#include "../utils/Target.hpp"
#include "../utils/Tiles.hpp"

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
  static DenseGemmTile generateDenseGemmTile(std::size_t num_ms, std::size_t dn_bw, std::size_t rn_bw, std::size_t M, std::size_t N, std::size_t K,
                                             Target target);

  // SparseDense
  static SparseDenseTile generateSparseDenseTile(std::size_t num_ms, std::size_t dn_bw, std::size_t rn_bw, std::size_t M, std::size_t N, std::size_t K,
                                                 float MK_sparsity, Target target);
};

}  // namespace StonneMapper

#endif  //STONNE_TILEGENERATOR_STONNEMAPPER_H
