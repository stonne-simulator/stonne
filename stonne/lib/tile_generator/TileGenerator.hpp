#ifndef STONNE_TILEGENERATOR_HPP
#define STONNE_TILEGENERATOR_HPP

#include "tile_generator/tile.hpp"
#include "tile_generator/types.hpp"

namespace TileGenerator {

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
  std::size_t num_ms;
  std::size_t dn_bw;
  std::size_t rn_bw;
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
  ConvTile generateConvTile(std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X, std::size_t Y,
                            std::size_t X_, std::size_t Y_, std::size_t stride, Target target);

  // DenseGEMM/FC
  DenseGemmTile generateDenseGemmTile(std::size_t M, std::size_t N, std::size_t K, Target target);

  // SparseDense
  SparseDenseTile generateSparseDenseTile(std::size_t M, std::size_t N, std::size_t K, float MK_sparsity, Target target);
};

}  // namespace TileGenerator

#endif  //STONNE_TILEGENERATOR_HPP