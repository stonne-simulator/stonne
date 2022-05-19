#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include <set>
#include "TileGenerator/StonneMapper/utility.h"

namespace StonneMapper {

    /*****************/
    /*** Constants ***/
    /*****************/

    // StonneMapper report base filename
    const std::string stonneMapperOutputBasename = "StonneMapper_output_";

    
    /************************/
    /*** Helper functions ***/
    /************************/

    // Calculates the edge utilization for a given dimension and its tile value
    // Edge utilization in a dimension is the percentage of operations that are worth for a given map
    float getEdgeUtilization(uint X, uint T_X) {
        float fX = float(X);
        float fT_X = float(T_X);
        return fX / (ceil(fX / fT_X) * fT_X);
    }


    // Gets all the divisors of a number
    std::set <uint> getAllDivisors(uint n) {
        std::set <uint> divisors;
        for (uint i = 1; i <= sqrt(n) + 1; i++) {
            if (n % i == 0) {
                divisors.insert(i);
                divisors.insert(n / i);
            }
        }
        return divisors;
    }


    // Adds to options all the possible tiles (using values that are powers of two) that follows a group of restrictions
    void addTilesPowersOfTwo(std::set<TileOptionDenseGemm> &options, uint M, uint N, uint K, uint num_ms) {
        for (unsigned long long int T_M = 1; T_M <= M; T_M *= 2) {
            for (unsigned long long int T_N = 1; T_N <= N; T_N *= 2) {
                for (unsigned long long int T_K = (K == 1 ? 1 : 2); T_K <= K; T_K *= 2) {
                    // We increase higher limits. For example, if M=127 then we could use T_M=128
                    // as tile. For this, we increase the tile options
                    // Also, we have to ensure that the tile used does not exceed the number of MS
                    if (T_M <= M * 2 && T_N <= N * 2 && T_K <= K * 2 && T_M * T_N * T_K <= num_ms) {
                        TileOptionDenseGemm option(num_ms, M, N, K, T_M, T_N, T_K);
                        options.insert(option);
                    }
                }
            }
        }
    }


    // Adds to options all the possible tiles that follows a group of restrictions
    // Note that the order of complexity would be very high if we considered all numbers. Therefore, we will only
    // select the dividing numbers of both each dimension and each dimension-1
    // However, this implementation should be improved and optimized if it is to be used for a very large size
    void addTilesUsingDivisors(std::set<TileOptionDenseGemm> &options, uint M, uint N, uint K, uint num_ms,
                               float minimumMsUtilization, float minimumEdgeUtilization) {
        // Get all divisors of each dimension
        std::set<uint> mDivisors = getAllDivisors(M);
        mDivisors.merge(getAllDivisors(M - 1));
        std::set<uint> nDivisors = getAllDivisors(N);
        nDivisors.merge(getAllDivisors(N - 1));
        std::set<uint> kDivisors = getAllDivisors(K);
        kDivisors.merge(getAllDivisors(K - 1));
        // remove 1 from kDivisors if K!=1
        if (K != 1)
            kDivisors.erase(1);

        for (unsigned long long int mDivisor : mDivisors) {
            for (unsigned long long int nDivisor : nDivisors) {
                for (unsigned long long int kDivisor : kDivisors) {
                    // We preselect which tiles to add to the set to avoid high memory usage
                    // Calculate the ms_utilization of the tile
                    float msUtilization = (float) (mDivisor * nDivisor * kDivisor) / float(num_ms);

                    // Calculate the mean of the edge utilization of the tile
                    float edgeUtilization = getEdgeUtilization(M, mDivisor);
                    edgeUtilization += getEdgeUtilization(N, nDivisor);
                    edgeUtilization += getEdgeUtilization(K, kDivisor);
                    edgeUtilization /= 3.0f;

                    if (mDivisor * nDivisor * kDivisor <= num_ms && // does not exceed ms
                        msUtilization >= minimumMsUtilization && // ms utilization is high enough
                        edgeUtilization >= minimumEdgeUtilization && // edge utilization is high enough
                        mDivisor <= M * 2 && nDivisor <= N * 2 && kDivisor <= K * 2) { // does not exceed the limits of each dimension
                        // Builds the tile and inserts it
                        TileOptionDenseGemm option(num_ms, M, N, K, mDivisor, nDivisor, kDivisor);
                        options.insert(option);
                    }
                }
            }
        }
    }


    // Generates a filename for the results generation of a DenseGEMM/FC layer
    std::string getStonneMapperOutputFilename(std::string layerType, uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K) {
        std::stringstream ss;
        ss << stonneMapperOutputBasename << layerType << "_num_ms" << num_ms << "_dn_bw" << dn_bw << "_rn_bw" << rn_bw <<
           "_M" << M << "_N" << N << "_K" << K << ".txt";
        return ss.str();
    }


    // Writes the results of the analysis made by the simulator for a DenseGEMM/FC layer
    void writeDenseGemmGeneratorResults(std::set<TileOptionDenseGemm> &options, TileOptionDenseGemm &bestTile,
                                        uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K) {
        std::ofstream outputFile(getStonneMapperOutputFilename("FC", num_ms, dn_bw, rn_bw, M, N, K));

        // hardware and layer parameters
        outputFile << "### StonneMapper results" << std::endl;
        outputFile << "# Parameters:" << std::endl;
        outputFile << " - Layer Type: DenseGEMM/FC" << std::endl;
        outputFile << " - num_ms: " << num_ms << std::endl;
        outputFile << " - dn_bw: " << dn_bw << std::endl;
        outputFile << " - rn_bw: " << rn_bw << std::endl;
        outputFile << " - M: " << M << std::endl;
        outputFile << " - N: " << N << std::endl;
        outputFile << " - K: " << K << std::endl;

        // Generator statistics
        outputFile << "# Statistics:" << std::endl;
        outputFile << " - Total number of tiles analyzed: " << options.size() << std::endl;

        // best tile generated/found
        outputFile << "# Best tile generated: " << std::endl;
        outputFile << " - <T_M=" << bestTile.T_M << ", T_N=" << bestTile.T_N << ", T_K=" << bestTile.T_K << ">" << std::endl;
        outputFile << " - MS utilization: " << bestTile.msUtilization << std::endl;
        outputFile << " - Edge utilization: " << bestTile.edgeUtilization << std::endl;
        outputFile << " - Heuristic value: " << bestTile.heuristic << std::endl;

        // all generated tiles and its results
        int i = 0;
        for (TileOptionDenseGemm to : options) {
            i++;
            outputFile << "===================================================" << std::endl;
            outputFile << "# [" << i << "]" << std::endl;
            outputFile << " - Tile: <T_M=" << to.T_M << ", T_N=" << to.T_N << ", T_K=" << to.T_K << ">" << std::endl;
            outputFile << " - MS utilization: " << to.msUtilization << std::endl;
            outputFile << " - Edge utilization: " << to.edgeUtilization << std::endl;
            outputFile << " - Heuristic value: " << to.heuristic << std::endl;
        }

        outputFile.close();
    }

    // Writes the results of the analysis made by the simulator for a SparseDense layer
    void writeSparseDenseGeneratorResults(std::set<TileOptionSparseDense> &options, TileOptionSparseDense &bestTile,
                                          uint num_ms, uint dn_bw, uint rn_bw, uint M, uint N, uint K, float sparsity) {
        std::ofstream outputFile(getStonneMapperOutputFilename("SparseDense", num_ms, dn_bw, rn_bw, M, N, K));

        // hardware and layer parameters
        outputFile << "### StonneMapper results" << std::endl;
        outputFile << "# Parameters:" << std::endl;
        outputFile << " - Layer Type: SparseDense" << std::endl;
        outputFile << " - num_ms: " << num_ms << std::endl;
        outputFile << " - dn_bw: " << dn_bw << std::endl;
        outputFile << " - rn_bw: " << rn_bw << std::endl;
        outputFile << " - M: " << M << std::endl;
        outputFile << " - N: " << N << std::endl;
        outputFile << " - K: " << K << std::endl;
        outputFile << " - Sparsity: " << sparsity << std::endl;

        // Generator statistics
        outputFile << "# Statistics:" << std::endl;
        outputFile << " - Total number of tiles analyzed: " << options.size() << std::endl;

        // best tile generated/found
        outputFile << "# Best tile generated: " << std::endl;
        outputFile << " - <T_N=" << bestTile.T_N << ", T_K=" << bestTile.T_K << ">" << std::endl;
        outputFile << " - MS utilization: " << bestTile.msUtilization << std::endl;
        outputFile << " - Edge utilization: " << bestTile.edgeUtilization << std::endl;
        outputFile << " - Heuristic value: " << bestTile.heuristic << std::endl;

        // all generated tiles and its results
        int i = 0;
        for (TileOptionSparseDense to : options) {
            i++;
            outputFile << "===================================================" << std::endl;
            outputFile << "# [" << i << "]" << std::endl;
            outputFile << " - Tile: <T_N=" << to.T_N << ", T_K=" << to.T_K << ">" << std::endl;
            outputFile << " - MS utilization: " << to.msUtilization << std::endl;
            outputFile << " - Edge utilization: " << to.edgeUtilization << std::endl;
            outputFile << " - Heuristic value: " << to.heuristic << std::endl;
        }

        outputFile.close();
    }

} // namespace StonneMapper