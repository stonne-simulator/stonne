#include <iostream>
#include <cassert>
#include "TileGenerator/TileGenerator.h"
#include "TileGenerator/mRNA/MrnaGenerator.h"
#include "TileGenerator/StonneMapper/StonneMapperGenerator.h"

namespace TileGenerator {

    /******************************/
    /*** Tile Generator Methods ***/
    /******************************/

    ConvTile TileGenerator::generateConvTile(uint R, uint S, uint C, uint K, uint G, uint N, uint X, uint Y, uint X_, uint Y_,
                              uint stride, Target target) {
        switch(generator) {
            case CHOOSE_AUTOMATICALLY: // by default: mRNA tool
            case MRNA:
                std::cout << "Using mRNA as Tile Generator for CONV layer" << std::endl;
                return mRNA::MrnaGenerator::generateConvTile(num_ms, dn_bw, rn_bw, R, S, C, K, G, N, X, Y, X_, Y_, stride, target);
            default:
                std::cerr << "Only mRNA generator is supported for CONV layers" << std::endl;
                assert(false);
        }
    }


    DenseGemmTile TileGenerator::generateDenseGemmTile(uint M, uint N, uint K, Target target) {
        switch(generator) {
            case MRNA:
                std::cout << "Using mRNA as Tile Generator for DenseGemm/FC layer" << std::endl;
                return mRNA::MrnaGenerator::generateDenseGemmTile(num_ms, dn_bw, rn_bw, M, N, K, target);
            case CHOOSE_AUTOMATICALLY: // by default: StonneMapper tool
            case STONNE_MAPPER:
                std::cout << "Using StonneMapper as Tile Generator for DenseGemm/FC layer" << std::endl;
                return StonneMapper::StonneMapperGenerator::generateDenseGemmTile(num_ms, dn_bw, rn_bw, M, N, K, target);
            default:
                std::cerr << "Only mRNA and StonneMapper generator is supported for DenseGemm/FC layers" << std::endl;
                assert(false);
        }
    }


    SparseDenseTile TileGenerator::generateSparseDenseTile(uint M, uint N, uint K, float MK_sparsity, Target target) {
        switch(generator) {
            case CHOOSE_AUTOMATICALLY: // by default: StonneMapper tool
            case STONNE_MAPPER:
                std::cout << "Using StonneMapper as Tile Generator for SparseDense layer" << std::endl;
                return StonneMapper::StonneMapperGenerator::generateSparseDenseTile(num_ms, dn_bw, rn_bw, M, N, K, MK_sparsity, target);
            default:
                std::cerr << "Only StonneMapper generator is supported for SparseDense layers" << std::endl;
                assert(false);
        }
    }

} // TileGenerator