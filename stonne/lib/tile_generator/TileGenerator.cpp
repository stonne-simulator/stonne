#include "TileGenerator.hpp"
#include <cassert>
#include <iostream>
#include "mRNA/MrnaGenerator.hpp"
#include "stonne_mapper/StonneMapperGenerator.hpp"

namespace TileGenerator {

/******************************/
/*** Tile Generator Methods ***/
/******************************/

ConvTile TileGenerator::generateConvTile(std::size_t R, std::size_t S, std::size_t C, std::size_t K, std::size_t G, std::size_t N, std::size_t X, std::size_t Y,
                                         std::size_t X_, std::size_t Y_, std::size_t stride, Target target) {
  switch (generator) {
    case CHOOSE_AUTOMATICALLY:  // by default: mRNA tool
    case MRNA:
      std::cout << "Using mRNA as Tile Generator for CONV layer" << std::endl;
      return mRNA::MrnaGenerator::generateConvTile(num_ms, dn_bw, rn_bw, R, S, C, K, G, N, X, Y, X_, Y_, stride, target);
    default:
      std::cerr << "Only mRNA generator is supported for CONV layers" << std::endl;
      assert(false);
  }
}

DenseGemmTile TileGenerator::generateDenseGemmTile(std::size_t M, std::size_t N, std::size_t K, Target target) {
  switch (generator) {
    case MRNA:
      std::cout << "Using mRNA as Tile Generator for DenseGemm/FC layer" << std::endl;
      return mRNA::MrnaGenerator::generateDenseGemmTile(num_ms, dn_bw, rn_bw, M, N, K, target);
    case CHOOSE_AUTOMATICALLY:  // by default: StonneMapper tool
    case STONNE_MAPPER:
      std::cout << "Using StonneMapper as Tile Generator for DenseGemm/FC layer" << std::endl;
      return StonneMapper::StonneMapperGenerator::generateDenseGemmTile(num_ms, dn_bw, rn_bw, M, N, K, target);
    default:
      std::cerr << "Only mRNA and StonneMapper generator is supported for DenseGemm/FC layers" << std::endl;
      assert(false);
  }
}

SparseDenseTile TileGenerator::generateSparseDenseTile(std::size_t M, std::size_t N, std::size_t K, float MK_sparsity, Target target) {
  switch (generator) {
    case CHOOSE_AUTOMATICALLY:  // by default: StonneMapper tool
    case STONNE_MAPPER:
      std::cout << "Using StonneMapper as Tile Generator for SparseDense layer" << std::endl;
      return StonneMapper::StonneMapperGenerator::generateSparseDenseTile(num_ms, dn_bw, rn_bw, M, N, K, MK_sparsity, target);
    default:
      std::cerr << "Only StonneMapper generator is supported for SparseDense layers" << std::endl;
      assert(false);
  }
}

}  // namespace TileGenerator