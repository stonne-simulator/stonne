#include <algorithm>
#include <set>
#include <cmath>
#include "TileGenerator/StonneMapper/StonneMapperGenerator.h"
#include "TileGenerator/StonneMapper/TileOptionDenseGemm.h"
#include "TileGenerator/StonneMapper/TileOptionSparseDense.h"
#include "TileGenerator/StonneMapper/utility.h"


namespace StonneMapper {

    /*****************/
    /*** Constants ***/
    /*****************/

    // Minimum MsUtilization and EdgeUtilization required for add a tile that is not power of two
    const float minimumMsUtilization = 0.8f;
    const float minimumEdgeUtilization = 0.8f;


    /**********************/
    /*** Helper Methods ***/
    /**********************/

    // Approximates to the best T_K value for a given sparsity grade
    float approximatePowerFactorT_K(float sparsity);


    /*******************************/
    /*** Tile Generation Methods ***/
    /*******************************/

    DenseGemmTile StonneMapperGenerator::generateDenseGemmTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, Target target) {
        /** This generation function does a search of combinations that maximizes the utilization of the MSs
         * 1. Add all combinations of powers of 2 (ms_utilization=1)
         * 2. Try to find out a group of combinations, not powers of 2, that surpass a certain threshold (ms_utilization>0.8)
         * 3. After that, we will calculate for each tile an heuristic value based on the next factors:
         * - The number of MSs that are used (ms_utilization)
         * - Mean of the M-N-K edges utilization (edge_utilization)
         * - Use of T_N and T_K
         * 4. Finally, we will select the tile with the highest heuristic value
         * - If two tiles have the same heuristic value (or very similar), we will select the one with higher T_N (and later T_K)
         *
         * Possible improvements:
         *  - Considerate dn_bw and rn_bw for the heuristic
         *  - Make different heuristics based on the specific target (performance (current), energy and energy-efficiency)
         */

        // 1. Add all combinations of powers of 2 (ms_utilization=1)
        std::set<TileOptionDenseGemm> options;
        addTilesPowersOfTwo(options, M, N, K, num_ms);

        // 2. Try to find out a group of combinations, not powers of 2, that surpass a certain threshold (ms_utilization>0.8)
        // For the moment, this function is enabled because using it we get worst results
        // The general heuristics of the generator must be improved, balancing the importance of each parameter,
        // so that this implementation can be useful
        //  addTilesUsingDivisors(options, M, N, K, num_ms, minimumMsUtilization, minimumEdgeUtilization);

        // 3. Get the best tile option generated based on its heuristic value
        TileOptionDenseGemm bestTile = *options.begin();
        for (TileOptionDenseGemm option : options) {
            if (option > bestTile) {
                bestTile = option;
            }
        }

        // 4. Prints the results of the analysis in the output file
        writeDenseGemmGeneratorResults(options, bestTile, num_ms, dn_bw, rn_bw, M, N, K);

        // 5. Return the best tile option
        return DenseGemmTile(bestTile.T_M, bestTile.T_N, bestTile.T_K);
    }


    SparseDenseTile StonneMapperGenerator::generateSparseDenseTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, float MK_sparsity, Target target) {
         /** Implement a solution which always obtain the optimum tile in this type of layers is difficult to design and
         * implement. For this, we have decide that the best way to implement this is only considering an approximation.
         * That's quite easy because in a big part of the cases the difference between two mappings tends to be so small,
         * like 0.90 speedup comparing with the best tile possible for this layer (this difference increases and maximizes
         * on the sparsity edge values [0, 1]).
         *
         * We can approach this generation in two ways:
         *  1) Approximate the value with a custom function which tries to simulate the evolution of the
         *     parameter T_K depending on the value of sparsity. We can get later the T_N value with num_ms / T_K
         *     At 0 sparsity T_K must be maximum (num_ms), while for 1 sparsity T_K must be minimum (K == 1 ? 1 : 2)
         *  2) Through an heuristic function that evaluates all the tiles that could be used, based on its MS utilization,
         *     edge utilization and its sparsity value. Although implement this heuristic its difficult because SparseDense
         *     layers does not follow a regular structure as DenseGEMM/FC layers, so it becomes more difficult to get
         *     good results using it.
         * Both ideas could be combined to reach a good solution, but at the moment that is not implemented because
         * results of the tests weren't good enough.
         *
         * In the first version of this implementation (19-05-2022) I have only considered to use the first option:
         * approximate the optimum tile through a function. In this case, I use a piecewise function that combines
         * exponential and linear parts (it will be explained in detail in its corresponding part. With this, I can
         * easily get the pseudo-optimum T_K value (and also the T_N) based on the sparsity grade.
         * Current problem of this implementation:
         *  - Works fine for matrix-dimensions that are powers of 2, but in other cases where the MS and edge utilization
         *    are not considered.
         *  - Also dn_bw and rn_bw are not considered and maybe could have impact in the final results.
         *  - Only supports target::Performance, but it can't consider another types of targets like energy.
         */

        std::set<TileOptionSparseDense> options;

        /* Implementation currently discarded because we don't consider the heuristic option
        // 1. Add all possible tiles for this layer (maximizing MS utilization)
        std::set<TileOptionSparseDense> options;
        for (uint T_K = (K == 1 ? 1 : 2); T_K <= K; T_K++) {
            uint T_N = std::min(N, num_ms / T_K);
            TileOptionSparseDense option(num_ms, M, N, K, T_N, T_K, MK_sparsity);
            options.insert(option);
        }
        */

        // Calculate the maximum T_K=2^x value we can use
        // We subtract 1 due to T_K tile constrains [min=1..max=512] -> [min=2..max=512] (min T_K=2)
        // Actually, for the final power we will increase the power by one
        uint maxPowerOfTwoT_K = log2(num_ms) - 1;

        // TODO: explain this correctly
        // If the value of N is much smaller than that of T_K, then the best results are obtained, by far, by increasing
        // the value of T_K. Therefore, depending on the difference between K and N, more priority will be given to T_K,
        // thus reducing the degree of sparsity to be considered.
        // E.g: K=4096, N=16, sparsity=0.9 -> log2(4096) - log(16) = 8 -> sparsity = 0.9 - (8 / 10) = 0.1 -> more priority to T_K
        if (N * 2 <= K)
            MK_sparsity = std::max(0.0, MK_sparsity - (log2(K) - log2(N)) / 10.0);

        // Approximate the power of 2 of T_K using the approximation function -> it returns a value between 0 and 1
        // with the factor/position of the power to select
        // E.g: num_ms=64 -> maxPowerOfTwoT_K=5 ; sparsity=0.6 -(suppose)-> func(sparsity)=0.4 ;
        // => assignedPower = maxPowerOfTwoT_K * sparsity = 2 ==> T_K=2^2=4
        uint power = round(maxPowerOfTwoT_K * approximatePowerFactorT_K(MK_sparsity));
        // Take care with the last subtract! Consider the K==1 -> T_K=1 case
        if (K==1) power = 0;
        else power++; // adds again the value, minimizing the minimum possible power to 1 (T_K>=2^1)

        // Get T_K using the calculated power of 2
        uint T_K = std::min(K, (uint) pow(2, power));
        // Assign the rest of the num_ms to T_N
        uint T_N = std::min(N, num_ms / T_K);
        // If after this there are still some multipliers left (e.g: N very small) then add them to T_K if possible
        T_K = std::min(K, T_K * (num_ms / (T_K * T_N)));


        // We implement this by this way to reuse the implementation of the heuristic method, thus it is not necessary
        // to modify the rest of the code of the function, and it is easily adaptable if you want to adapt the new
        // strategy (either to replace it or combine it)
        TileOptionSparseDense approximationFunctionTile(num_ms, M, N, K, T_N, T_K, MK_sparsity);
        options.insert(approximationFunctionTile);


        // 2. Get the best tile option generated based on its heuristic value
        TileOptionSparseDense bestTile = *options.begin();
        for (TileOptionSparseDense option : options) {
            if (option > bestTile) {
                bestTile = option;
            }
        }

        // 3. Prints the results of the analysis in the output file
        writeSparseDenseGeneratorResults(options, bestTile, num_ms, dn_bw, rn_bw, M, N, K, MK_sparsity);

        // 4. Return the best tile option
        return SparseDenseTile(bestTile.T_N, bestTile.T_K);
    }


    /**********************/
    /*** Helper Methods ***/
    /**********************/

    // Approximates to the best T_K value for a given sparsity grade
    // Returns a factor value between 0 (T_K=max_value) and 1 (T_K=min_value=2)
    float approximatePowerFactorT_K(float sparsity) {
        // Approximate point of intersection of the two functions
        if (sparsity < 0.51327166727) {
            // Basic exponential decreasing function
            // Generated with https://www.mycurvefit.com/ and this data:
            // x     y
            // ==========
            // 0     1
            // 0.05  0.8
            // 0.1   0.6
            // 0.5   0.5
            return 0.4855423 + 0.5192579 * exp(-11.84906 * sparsity);
        }
        else {
            // Linear decreasing function
            return 1 - sparsity;
        }
    }

} // namespace StonneMapper