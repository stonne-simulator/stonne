#ifndef STONNE_TILEGENERATOR_TARGET_H
#define STONNE_TILEGENERATOR_TARGET_H

#include "TileGenerator/mRNA/define.h"

namespace TileGenerator {

    enum Target {
        NONE = mRNA::none,
        PERFORMANCE = mRNA::performance,
        ENERGY = mRNA::energy,
        ENERGY_EFFICIENCY = mRNA::energy_efficiency
    };

} // namespace TileGenerator

#endif //STONNE_TILEGENERATOR_TARGET_H
