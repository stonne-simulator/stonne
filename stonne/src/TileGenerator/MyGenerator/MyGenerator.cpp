#include <algorithm>
#include <cmath>
#include "utility.h"
#include "TileGenerator/MyGenerator/MyGenerator.h"


namespace MyGenerator {

    /*****************/
    /*** Constants ***/
    /*****************/
    const float minimumEdgeUtilizationPercentage = 0.8f;
    const float edgeUtilizationEpsilon = 0.0001f;


    /*******************************/
    /*** Tile Generation Methods ***/
    /*******************************/

    DenseGemmTile MyGenerator::generateDenseGemmTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, Target target) {
        // TODO: consider variations between targets

        uint T_N = std::min(nextPowerOf2(N), K > 1 ? num_ms / 2 : num_ms);
        uint T_K = std::min(nextPowerOf2(K), num_ms / T_N);
        uint T_M = std::min(nextPowerOf2(M), num_ms / (T_N * T_K));

        maximizeEdgeUtilization(M, N, K, T_M, T_N, T_K);

        std::cout << "Total Occupancy of MS: " << float(T_M * T_N * T_K) * 100 / float(num_ms) << "%" << std::endl;
        std::cout << "Total Useful Operations in M: " << getEdgeUtilization(M, T_M) << "%" << std::endl;
        std::cout << "Total Useful Operations in N: " << getEdgeUtilization(N, T_N) << "%" << std::endl;
        std::cout << "Total Useful Operations in K: " << getEdgeUtilization(K, T_K) << "%" << std::endl;

        return DenseGemmTile(T_M, T_N, T_K);
    }


    SparseDenseTile MyGenerator::generateSparseDenseTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, float MK_sparsity, Target target) {
        // TODO: implement this
        return SparseDenseTile(0, 0);
    }


    /************************/
    /*** Helper functions ***/
    /************************/

    float MyGenerator::getEdgeUtilization(uint X, uint T_X) {
        return X * 100 / (ceil((float) X / (float) T_X) * T_X);
    }


    void MyGenerator::maximizeEdgeUtilization(uint M, uint N, uint K, uint &T_M, uint &T_N, uint &T_K) {
        float utilization;
        while ((utilization = getEdgeUtilization(N, T_N)) < minimumEdgeUtilizationPercentage && T_K <= K) {
            // only apply the change if there will be any improvement
            if (abs(getEdgeUtilization(N, T_N / 2) - utilization) < edgeUtilizationEpsilon)
                break;

            T_N /= 2;
            T_K *= 2;
        }
        while ((utilization = getEdgeUtilization(N, T_N)) < minimumEdgeUtilizationPercentage && T_M <= M) {
            // only apply the change if there will be any improvement
            if (abs(getEdgeUtilization(N, T_N / 2) - utilization) < edgeUtilizationEpsilon)
                break;

            T_N /= 2;
            T_M *= 2;
        }
    }

} // namespace MyGenerator