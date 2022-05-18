#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>
#include "utility.h"
#include "TileGenerator/StonneMapper/StonneMapperGenerator.h"
#include "TileGenerator/StonneMapper/TileOption.h"


namespace StonneMapper {

    /*****************/
    /*** Constants ***/
    /*****************/

    // StonneMapper report base filename
    const std::string stonneMapperOutputBasename = "StonneMapper_output_";


    /************************/
    /*** Helper functions ***/
    /************************/

    std::string getStonneMapperOutputFilename(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K);
    void writeGeneratorResults(std::vector<TileOption> &options, TileOption &bestTile, std::string layerType, uint num_ms,
                               uint dn_bw, uint rn_bw, uint M, uint N, uint K, float sparsity=0.0f);


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
        std::vector<TileOption> options;
        uint minNumberIterations = std::numeric_limits<uint>::max();
        // TODO: increase limits * 2?
        for (unsigned long long int T_M = 1; T_M <= M; T_M *= 2) {
            for (unsigned long long int T_N = 1; T_N <= N; T_N *= 2) {
                for (unsigned long long int T_K = (K == 1 ? 1 : 2); T_K <= K; T_K *= 2) {
                    if (T_M <= M && T_N <= N && T_K <= K && T_M * T_N * T_K <= num_ms) {
                        TileOption option(num_ms, M, N, K, T_M, T_N, T_K);
                        options.push_back(option);
                    }
                }
            }
        }

        // 2. Try to find out a group of combinations, not powers of 2, that surpass a certain threshold (ms_utilization>0.8)
        // TODO: implement this

        // 3. Get the best tile option generated based on utilization, steps, etc
        TileOption bestTile = options.front();
        for (int i = 1; i < options.size(); i++) {
            if (options[i] > bestTile) {
                bestTile = options[i];
            }
        }

        // 4. Return the best tile option
        writeGeneratorResults(options, bestTile, "DenseGEMM/FC", num_ms, dn_bw, rn_bw, M, N, K);
        return DenseGemmTile(bestTile.T_M, bestTile.T_N, bestTile.T_K);
    }


    SparseDenseTile StonneMapperGenerator::generateSparseDenseTile(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, float MK_sparsity, Target target) {
        // TODO: implement this
        return SparseDenseTile(0, 0);
    }


    /************************/
    /*** Helper functions ***/
    /************************/

    // Generates a filename for the results generation of a DenseGEMM/FC layer
    std::string getStonneMapperOutputFilename(uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K) {
        std::stringstream ss;
        ss << stonneMapperOutputBasename << "FC" << "_num_ms" << num_ms << "_dn_bw" << dn_bw << "_rn_bw" << rn_bw <<
           "_M" << M << "_N" << N << "_K" << K << ".txt";
        return ss.str();
    }


    // Writes the results of the analysis made by the simulator
    void writeGeneratorResults(std::vector<TileOption> &options, TileOption &bestTile, std::string layerType, uint num_ms,
                               uint dn_bw, uint rn_bw, uint M, uint N, uint K, float sparsity) {
        std::ofstream outputFile(getStonneMapperOutputFilename(num_ms, dn_bw, rn_bw, M, N, K));

        // hardware and layer parameters
        outputFile << "### StonneMapper results" << std::endl;
        outputFile << "# Parameters:" << std::endl;
        outputFile << " - Layer Type: " << layerType << std::endl;
        outputFile << " - num_ms: " << num_ms << std::endl;
        outputFile << " - dn_bw: " << dn_bw << std::endl;
        outputFile << " - rn_bw: " << rn_bw << std::endl;
        outputFile << " - M: " << M << std::endl;
        outputFile << " - N: " << N << std::endl;
        outputFile << " - K: " << K << std::endl;

        // best tile generated/found
        outputFile << "# Best tile generated: " << std::endl;
        outputFile << " - <T_M=" << bestTile.T_M << ", T_N=" << bestTile.T_N << ", T_K=" << bestTile.T_K << ">" << std::endl;
        outputFile << " - MS utilization: " << bestTile.msUtilization << std::endl;
        outputFile << " - Edge utilization: " << bestTile.edgeUtilization << std::endl;
        outputFile << " - Heuristic value: " << bestTile.heuristic << std::endl;

        // all generated tiles and its results
        for (int i = 0; i < options.size(); i++) {
            TileOption to = options[i];
            outputFile << "===================================================" << std::endl;
            outputFile << "# [" << i << "]" << std::endl;
            outputFile << " - Tile: <T_M=" << to.T_M << ", T_N=" << to.T_N << ", T_K=" << to.T_K << ">" << std::endl;
            outputFile << " - MS utilization: " << to.msUtilization << std::endl;
            outputFile << " - Edge utilization: " << to.edgeUtilization << std::endl;
            outputFile << " - Heuristic value: " << to.heuristic << std::endl;
        }

        outputFile.close();
    }

} // namespace StonneMapper