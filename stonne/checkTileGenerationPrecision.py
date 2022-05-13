import argparse
from calendar import c
from math import log2
import os
import re

DEFAULT_TOLERANCE = 10
SPARSITY_VALUES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99]


def test(num_ms, dn_bw, rn_bw, M, N, K, tolerance):
    tileGen_results = []
    search_result = []

    # all executions, obtaining the number of cycles of each configuration
    for sparsity in SPARSITY_VALUES:
        # execution generating a new tile
        output = os.popen(f'./stonne -SparseDense -M={M} -N={N} -K={K} -MK_sparsity={sparsity} \
                            -num_ms={num_ms} -dn_bw={dn_bw} -rn_bw={rn_bw} -accumulation_buffer=1 \
                            -T_K=1 -T_N=1').read()#-generate_tile=performance').read()
        # TODO: get too the generated tile
        tile = [1, 1] # get_tile(output)
        cycles = get_cycles(output)
        tileGen_results.append([cycles, tile])
        if cycles == 0:
            print('ERROR: Tile generation failed for sparsity {}'.format(sparsity))
            
        # search for best mapping trying all combinations
        best_cycles = 999999999
        best_tile = ['-', '-']
        for T in [2**i for i in range(0, int(log2(num_ms))+1)]:
            T_K = T
            T_N = int(num_ms / T)

            output = os.popen(f'./stonne -SparseDense -M={M} -N={N} -K={K} -MK_sparsity={sparsity} \
                    -num_ms={num_ms} -dn_bw={dn_bw} -rn_bw={rn_bw} -accumulation_buffer=1 \
                    -T_K={T_K} -T_N={T_N}').read()
            cycles = get_cycles(output)
            if cycles > 0 and cycles < best_cycles:
                best_cycles = cycles
                best_tile = [T_K, T_N]
        search_result.append([best_cycles, best_tile])
         
    # print results as a csv
    print_results_csv(tileGen_results, search_result, tolerance)


def get_cycles(output):
    match = re.search(r'Number of cycles running: (\d+)', output)
    if match:
        return int(match.group(1))
    else:
        return 0


def get_tile(output):
    match = re.search(r'Generated Tile:.*T_N=(\d+).*T_K=(\d+)', output)
    if match:
        return int(match.group(1)), int(match.group(2)) # TODO: review this
    else:
        return 0


def print_results_csv(tileGen, search, tolerance):
    success_tiles = len(SPARSITY_VALUES)
    success_cases = 0
    improvements = 0
    
    print(f'Sparsity,tileGen_cycles,tileGen_T_K,tileGen_T_N,Search_cycles,Search_T_K,Search_T_N,Speedup')
    for i in range(len(SPARSITY_VALUES)):
        sparsity = SPARSITY_VALUES[i]
        tileGen_cycles = tileGen[i][0] if tileGen[i][0] > 0 else 'Error'
        tileGen_T_K = tileGen[i][1][0]
        tileGen_T_N = tileGen[i][1][1]
        search_cycles = search[i][0]
        search_T_K = search[i][1][0]
        search_T_N = search[i][1][1]
        speedup = ((1/tileGen_cycles) / (1/search_cycles)) if tileGen_cycles != 'ERROR' else '-'
        print(f'{sparsity},{tileGen_cycles},{tileGen_T_K},{tileGen_T_N},{search_cycles},{search_T_K},{search_T_N},{speedup}')
        
        if tileGen_cycles == 'Error':
            success_tiles -= 1
        if speedup >= 1: # best tile found
            improvements += 1
            success_cases += 1
        elif abs(speedup - 1) * 100 <= tolerance: # accepted tile for the given tolerance
            success_cases += 1

    print(f'\nTotal success tile generations: {success_tiles} / {len(SPARSITY_VALUES)}')
    print(f'Total tiles that improves performance or generates the best possible tile: {improvements} / {len(SPARSITY_VALUES)}')
    print(f'Total tiles that achieves the required precision ({tolerance}%): {success_cases} / {len(SPARSITY_VALUES)}')

if __name__ == "__main__":

    # argument parsing
    parser = argparse.ArgumentParser(description='Check the precision and correction of the tile generation.')
    parser.add_argument('--num_ms', type=int, required=True, help='Number of multipler switches.')
    parser.add_argument('--dn_bw', type=int, required=True, help='Bandwidth of the distribution network.')
    parser.add_argument('--rn_bw', type=int, required=True, help='Bandwidth of the reduction network.')
    parser.add_argument('--M', type=int, required=True, help='Matrix M size.')
    parser.add_argument('--N', type=int, required=True, help='Matrix N size.')
    parser.add_argument('--K', type=int, required=True, help='Matrix K size.')
    parser.add_argument('--tolerance', type=int, default=DEFAULT_TOLERANCE, help='Tolerance for deviation on the results.')
    args = parser.parse_args()

    print('\nParameters used:')
    print(f'\tnum_ms: {args.num_ms}')
    print(f'\tdn_bw: {args.dn_bw}')
    print(f'\trn_bw: {args.rn_bw}')
    print(f'\tM: {args.M}')
    print(f'\tN: {args.N}')
    print(f'\tK: {args.K}')
    print(f'\ttolerance: {args.tolerance}')

    os.system('make -j8 all')

    test(args.num_ms, args.dn_bw, args.rn_bw, args.M, args.N, args.K, args.tolerance)

    os.system('rm output_stats*')
