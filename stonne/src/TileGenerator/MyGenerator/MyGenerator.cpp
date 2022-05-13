#include <algorithm>
#include <cmath>
#include "utility.h"
#include "TileGenerator/MyGenerator/MyGenerator.h"

#define MINIMUM_PERCENTAGE_OPERATIONS 0.8f
#define UTILIZATION_EPSILON 0.0001f

namespace MyGenerator {

    float calculateUtilization(uint X, uint T_X) {
        return X * 100 / (ceil((float) X / (float) T_X) * T_X);
    }

    void maximizeUsefulOperations(uint M, uint N, uint K, uint &T_M, uint &T_N, uint &T_K) {
        float utilization;
        while ((utilization = calculateUtilization(N, T_N)) < MINIMUM_PERCENTAGE_OPERATIONS && T_K <= K) {
            // only apply the change if there will be any improvement
            if (abs(calculateUtilization(N, T_N / 2) - utilization) < UTILIZATION_EPSILON)
                break;

            T_N /= 2;
            T_K *= 2;
        }
        while ((utilization = calculateUtilization(N, T_N)) < MINIMUM_PERCENTAGE_OPERATIONS && T_M <= M) {
            // only apply the change if there will be any improvement
            if (abs(calculateUtilization(N, T_N / 2) - utilization) < UTILIZATION_EPSILON)
                break;

            T_N /= 2;
            T_M *= 2;
        }
    }

    DenseGemmTile MyGenerator::generateDenseGemmTile(uint M, uint N, uint K, Target target) {
        // TODO: consider variations between targets

        uint T_N = std::min(nextPowerOf2(N), K > 1 ? num_ms / 2 : num_ms);
        uint T_K = std::min(nextPowerOf2(K), num_ms / T_N);
        uint T_M = std::min(nextPowerOf2(M), num_ms / (T_N * T_K));

        maximizeUsefulOperations(M, N, K, T_M, T_N, T_K);

        std::cout << "Total Occupancy of MS: " << float(T_M * T_N * T_K) * 100 / float(num_ms) << "%" << std::endl;
        std::cout << "Total Useful Operations in M: " << calculateUtilization(M, T_M) << "%" << std::endl;
        std::cout << "Total Useful Operations in N: " << calculateUtilization(N, T_N) << "%" << std::endl;
        std::cout << "Total Useful Operations in K: " << calculateUtilization(K, T_K) << "%" << std::endl;

        return DenseGemmTile(T_M, T_N, T_K);
    }

    SparseDenseTile MyGenerator::generateSparseDenseTile(uint M, uint N, uint K, float MK_sparsity) {
        return SparseDenseTile(0, 0);
    }
} // namespace MyGenerator