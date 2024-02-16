#ifndef STONNE_TYPES_HPP
#define STONNE_TYPES_HPP

#include "mRNA/define.hpp"

namespace TileGenerator {

/**
 * Currently supported generators (by default: CHOOSE_AUTOMATICALLY)
 */
enum Generator { CHOOSE_AUTOMATICALLY = 0, MRNA = 1, STONNE_MAPPER = 2 };

enum Target { NONE = mRNA::none, PERFORMANCE = mRNA::performance, ENERGY = mRNA::energy, ENERGY_EFFICIENCY = mRNA::energy_efficiency };

}  // namespace TileGenerator

#endif  //STONNE_TYPES_H
