#include <iostream>
#include <cmath>
#include "TileGenerator/StonneMapper/TileOptionSparseDense.h"
#include "TileGenerator/StonneMapper/utility.h"


namespace StonneMapper {

    TileOptionSparseDense::TileOptionSparseDense(uint num_ms, uint M, uint N, uint K, uint T_N, uint T_K, float sparsity) {
        // Check that tile does not exceed the value of each dimension
        this->T_N = std::min(N, T_N);
        this->T_K = std::min(K, T_K);

        // Calculate the utilization of the multiples switches for this tile
        this->msUtilization = double(T_N * T_K) / double(num_ms);

        // Calculate the edge utilization for this tile in each dimension and get the mean
        float partialEdgeN = getEdgeUtilization(N, T_N);
        float partialEdgeK = getEdgeUtilization(K, T_K);
        this->edgeUtilization = (partialEdgeN + partialEdgeK) / 2.0f;

        // Just add all the values together. The max value of the heuristic is 2 at the moment
        // It's the simplest strategy, but it's not very effective
        this->heuristic = this->msUtilization + this->edgeUtilization;
    }


    bool TileOptionSparseDense::operator==( const TileOptionSparseDense& val ) const {
        if (T_N != val.T_N) return false;
        if (T_K != val.T_K) return false;
        return true;
    }


    bool TileOptionSparseDense::operator<( const TileOptionSparseDense& val ) const {
        /* Not considered at the moment
        // if heuristics are enough similar...
        if (std::abs(heuristic - val.heuristic) < heuristicEpsilon) {
            // then compare by T_N->T_K->T_M
            if (T_N < val.T_N) return true;
            else if (T_N < val.T_N) return false;
            else {
                return T_K > val.T_K;
            }
        }
        else if (heuristic < val.heuristic) { // higher heuristic -> better tile
            return true;
        }
        return false;
        */
        return heuristic < val.heuristic;
    }


    bool TileOptionSparseDense::operator>( const TileOptionSparseDense& val ) const {
        /* Not considered at the moment
        // if heuristics are enough similar...
        if (std::abs(heuristic - val.heuristic) < heuristicEpsilon) {
            // then compare by T_N->T_K->T_M
            if (T_N > val.T_N) return true;
            else if (T_N < val.T_N) return false;
            else {
                return T_K > val.T_K;
            }
        }
        else if (heuristic > val.heuristic) { // higher heuristic -> better tile
            return true;
        }
        return false;
         */
        return heuristic > val.heuristic;
    }

} // namespace StonneMapper