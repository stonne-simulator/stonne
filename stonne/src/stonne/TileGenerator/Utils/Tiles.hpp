#ifndef STONNE_TILEGENERATOR_TILES_H
#define STONNE_TILEGENERATOR_TILES_H

#include <cstddef>

namespace TileGenerator {

class ConvTile {
 public:
  std::size_t T_R;
  std::size_t T_S;
  std::size_t T_C;
  std::size_t T_K;
  std::size_t T_G;
  std::size_t T_N;
  std::size_t T_X_;
  std::size_t T_Y_;

  ConvTile(std::size_t T_R, std::size_t T_S, std::size_t T_C, std::size_t T_K, std::size_t T_G, std::size_t T_N, std::size_t T_X_, std::size_t T_Y_)
      : T_R(T_R), T_S(T_S), T_C(T_C), T_K(T_K), T_G(T_G), T_N(T_N), T_X_(T_X_), T_Y_(T_Y_) {}

  ~ConvTile() = default;
};

class DenseGemmTile {
 public:
  std::size_t T_M;
  std::size_t T_N;
  std::size_t T_K;

  DenseGemmTile(std::size_t T_M, std::size_t T_N, std::size_t T_K) : T_M(T_M), T_N(T_N), T_K(T_K) {}

  ~DenseGemmTile() = default;
};

class SparseDenseTile {
 public:
  std::size_t T_N;
  std::size_t T_K;

  SparseDenseTile(std::size_t T_N, std::size_t T_K) : T_N(T_N), T_K(T_K) {}

  ~SparseDenseTile() = default;
};

}  // namespace TileGenerator

#endif  //STONNE_TILEGENERATOR_TILES_H
