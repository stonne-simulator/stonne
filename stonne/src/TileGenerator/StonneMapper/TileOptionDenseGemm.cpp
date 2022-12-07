#include <iostream>
#include <cmath>
#include "TileGenerator/StonneMapper/TileOptionDenseGemm.h"
#include "TileGenerator/StonneMapper/utility.h"


namespace StonneMapper {

    TileOptionDenseGemm::TileOptionDenseGemm(uint num_ms, uint M, uint N, uint K, uint T_M, uint T_N, uint T_K) {
        // Check that tile does not exceed the value of each dimension
        this->T_M = std::min(M, T_M);
        this->T_N = std::min(N, T_N);
        this->T_K = std::min(K, T_K);

        // Calculate the utilization of the multiples switches for this tile
        this->msUtilization = double(T_M * T_N * T_K) / double(num_ms);

        // Calculate the edge utilization for this tile in each dimension and get the mean
        float partialEdgeM = getEdgeUtilization(M, T_M);
        float partialEdgeN = getEdgeUtilization(N, T_N);
        float partialEdgeK = getEdgeUtilization(K, T_K);
        this->edgeUtilization = (partialEdgeM + partialEdgeN + partialEdgeK) / 3.0f;

        // Calculate the heuristic value for this tile
        // - Max value added is, for T_N, factorHeuristicN. This is because usually tiles that maximizes the T_N value
        //   tends to get better results. Also, it happens with the performance when we compare T_K with T_M. For this
        //   reason, T_M does not receive any bonus if it gets a higher value
        // - Based on the max value that the dimension T_N/T_K can get, we calculate the "extra bonus" to the heuristic
        //   (to give this tiles more priority) based on the utilization on its dimension
        // - Note that tiles with higher T_N and T_K don't always have to be the bests. Realize that, when bigger is the
        //   T_N/T_K value, the getEdgeUtilization tends to be lower in the majority of cases. This increment is to
        //   balance the heuristic because the importance of keep this values higher
        float heuristicN = std::min(factorHeuristicN, (float(T_N) / std::min(float(N), float(num_ms))) * factorHeuristicN);
        float heuristicK = std::min(factorHeuristicK, (float(T_K) / std::min(float(K), float(num_ms))) * factorHeuristicK);
        // Just add all the values together. The max value of the heuristic is 2.15
        this->heuristic = this->msUtilization + this->edgeUtilization + heuristicN + heuristicK;
    }


    bool TileOptionDenseGemm::operator==( const TileOptionDenseGemm& val ) const {
        if (T_M != val.T_M) return false;
        if (T_N != val.T_N) return false;
        if (T_K != val.T_K) return false;
        return true;
    }


    bool TileOptionDenseGemm::operator<( const TileOptionDenseGemm& val ) const {
        // if heuristics are enough similar...
        if (std::abs(heuristic - val.heuristic) < heuristicEpsilon) {
            // then compare by T_N->T_K->T_M
            if (T_N < val.T_N) return true;
            else if (T_N < val.T_N) return false;
            else {
                if (T_K < val.T_K) return true;
                else if (T_K > val.T_K) return false;
                else return T_M < val.T_M;
            }
        }
        else if (heuristic < val.heuristic) { // higher heuristic -> better tile
            return true;
        }
        return false;
    }


    bool TileOptionDenseGemm::operator>( const TileOptionDenseGemm& val ) const {
        // if heuristics are enough similar...
        if (std::abs(heuristic - val.heuristic) < heuristicEpsilon) {
            // then compare by T_N->T_K->T_M
            if (T_N > val.T_N) return true;
            else if (T_N < val.T_N) return false;
            else {
                if (T_K > val.T_K) return true;
                else if (T_K < val.T_K) return false;
                else return T_M > val.T_M;
            }
        }
        else if (heuristic > val.heuristic) { // higher heuristic -> better tile
            return true;
        }
        return false;
    }

} // namespace StonneMapper