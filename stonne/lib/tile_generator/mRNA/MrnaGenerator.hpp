#ifndef STONNE_TILEGENERATOR_MRNAGENERATOR_H
#define STONNE_TILEGENERATOR_MRNAGENERATOR_H

#include "tile_generator/mRNA/Analyzer.hpp"
#include "tile_generator/tile.hpp"
#include "tile_generator/types.hpp"

using namespace TileGenerator;

namespace mRNA {

/**
 * MrnaGenerator abstracts the generation of tiles using the mRNA tool.
 * At the moment, mRNA only supports the generation of CONV and DenseGEMM/FC layers.
 */
class MrnaGenerator {
 public:
  /*******************************/
  /*** Tile Generation Methods ***/
  /*******************************/

  // CONV
  static ConvTile generateConvTile(std::size_t num_ms, std::size_t dn_bw, std::size_t rn_bw, std::size_t R, std::size_t S, std::size_t C, std::size_t K,
                                   std::size_t G, std::size_t N, std::size_t X, std::size_t Y, std::size_t X_, std::size_t Y_, std::size_t stride,
                                   Target target);

  // DenseGEMM/FC
  static DenseGemmTile generateDenseGemmTile(std::size_t num_ms, std::size_t dn_bw, std::size_t rn_bw, std::size_t M, std::size_t N, std::size_t K,
                                             Target target);

 private:
  /*************************************/
  /*** Helper functions declarations ***/
  /*************************************/

  // mRNA model creation
  static DNNModel createDNNModel(std::string layer_type, std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N,
                                 std::size_t X, std::size_t Y, std::size_t X_, std::size_t Y_, std::size_t stride);
  static DNNModel createConvModel(std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X, std::size_t Y,
                                  std::size_t X_, std::size_t Y_, std::size_t stride);
  static DNNModel createDenseGemmModel(std::size_t M, std::size_t N, std::size_t K);

  // output filename generation
  static std::string getMrnaOutputFilename(std::size_t num_ms, std::size_t dn_bw, std::size_t rn_bw, std::size_t R, std::size_t S, std::size_t C, std::size_t K,
                                           std::size_t G, std::size_t N, std::size_t X, std::size_t Y, std::size_t X_, std::size_t Y_, std::size_t stride);
  static std::string getMrnaOutputFilename(std::size_t num_ms, std::size_t dn_bw, std::size_t rn_bw, std::size_t M, std::size_t N, std::size_t K);
};

}  // namespace mRNA

#endif  //STONNE_TILEGENERATOR_MRNAGENERATOR_H
