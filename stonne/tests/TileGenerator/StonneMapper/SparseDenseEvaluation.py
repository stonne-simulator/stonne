import argparse
from math import log2, ceil
import os
import subprocess
import sys
try: # Try to import the local version first (usually works when executed from the command line with Python directly)
    import EvaluationUtils
except ImportError: # Only works when you execute it with the '-m unittest' parameter from stonne/stonne directory
    import tests.TileGenerator.StonneMapper.EvaluationUtils as EvaluationUtils


DEFAULT_TOLERANCE = 0.1


def evaluate(testname, num_ms, dn_bw, rn_bw, M, N, K, sparsity, tolerance=DEFAULT_TOLERANCE, generator="StonneMapper"):
    print(f'### SparseDense evaluation of {generator} with M={M}, N={N}, K={K}, sparsity={sparsity} and tolerance={tolerance}')

    ### Results from the generator ###

    print('# Use generator to find an optimum tile')

    # execution generating a new tile automatically
    command = f'./stonne -SparseDense -M={M} -N={N} -K={K} -MK_sparsity={sparsity} -num_ms={num_ms} -dn_bw={dn_bw}'
    command += f' -rn_bw={rn_bw} -accumulation_buffer=1 -generate_tile=performance -generator={generator}'
    print(f' - {command}')
    process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    # get results from execution
    generatedtile_cycles = EvaluationUtils.get_cycles(stdout)
    if generatedtile_cycles == 0 or EvaluationUtils.check_assertions(stderr) or process.returncode != 0:
        print('There was an error during execution. Cause:', file=sys.stderr)
        print(stderr, file=sys.stderr)
        return False

    generatedtile = EvaluationUtils.get_sparsedense_tile(stdout)

    # print tile and cycles in terminal
    print(f'\tL=> T_N={generatedtile[0]} T_K={generatedtile[1]}')
    print(f'\tL=> {generatedtile_cycles} cycles')


    ### Search of the best mapping using powers of 2 ###

    # search for best mapping trying all combinations
    min_cycles = 9999999999
    min_tile = None
    print('# Trying all combinations searching the best tile using powers of 2')
    for T_N in [2 ** i for i in range(0, int(ceil(log2(num_ms))) + 1)]:
        for T_K in [2 ** i for i in range(0, int(ceil(log2(num_ms))) + 1)]:
            # ensure that num_ms occupation is maximum and tile size does not exceed its dimension,
            # although allowing to slightly exceed the limit (conditions on the end of the loops)
            # cases with T_K == 1 and K != 1 will not work, we have to ensure that only can T_K=1 when K=1
            # also we always want to maximize the num_ms utilization
            if (T_K == 1 and K > 1) or T_N * T_K != num_ms:
                continue

            # execution using a fixed tile
            command = f'./stonne -SparseDense -M={M} -N={N} -K={K} -MK_sparsity={sparsity} -num_ms={num_ms}'
            command += f' -dn_bw={dn_bw} -rn_bw={rn_bw} -accumulation_buffer=1 -T_N={T_N} -T_K={T_K}'
            print(f' - {command}')
            process = subprocess.Popen(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            stdout = stdout.decode('utf-8')
            stderr = stderr.decode('utf-8')

            # get results from execution
            cycles = EvaluationUtils.get_cycles(stdout)
            if generatedtile_cycles == 0 or EvaluationUtils.check_assertions(stderr) or process.returncode != 0:
                # if this execution failed, try the next combination
                continue

            # updates the best tile found
            if cycles < min_cycles:
                min_cycles = cycles
                min_tile = [T_N, T_K]

            # print cycles of this case in terminal
            print(f'\tL=> {cycles} cycles')

            # if T_K has reached the largest value that makes sense for the mapping, cancel the loop
            if T_K >= K * 2:
                break
        # if T_N has reached the largest value that makes sense for the mapping, cancel the loop
        if T_N >= N * 2:
            break

    # get and print final results
    speedup = ((1 / generatedtile_cycles) / (1 / min_cycles))
    passed = True if 1 - speedup <= tolerance else False
    print('# Final results for SparseDense layer')
    print(f' - Hardware parameters: num_ms={num_ms}, dn_bw={dn_bw}, rn_bw={rn_bw}')
    print(f' - Matrix sizes: M={M}, N={N}, K={K}')
    print(f' - Tile Generated Automatically: T_N={generatedtile[0]}, T_K={generatedtile[1]}')
    print(f' - Total Cycles for generated tile: {generatedtile_cycles}')
    print(f' - Best tile found: T_N={min_tile[0]}, T_K={min_tile[1]}')
    print(f' - Total Cycles for best tile: {min_cycles}')
    print(f' - Speedup of the generated tile: {speedup}')
    print(f' - Pass the test (tolerance={tolerance})? {passed}')

    EvaluationUtils.save_sparsedense_results_csv(passed, testname, num_ms, dn_bw, rn_bw, M, N, K, sparsity, generator, generatedtile,
                                                 generatedtile_cycles, min_tile, min_cycles, speedup, tolerance)

    return passed


# Main method to execute a custom testcase
if __name__ == "__main__":
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this test script from the stonne/stonne directory")
        exit(1)

    # argument parsing
    parser = argparse.ArgumentParser(description='Check the precision and correction of the tile generation.')
    parser.add_argument('--num_ms', type=int, required=True, help='Number of multipler switches.')
    parser.add_argument('--dn_bw', type=int, required=True, help='Bandwidth of the distribution network.')
    parser.add_argument('--rn_bw', type=int, required=True, help='Bandwidth of the reduction network.')
    parser.add_argument('--M', type=int, required=True, help='Matrix M size.')
    parser.add_argument('--N', type=int, required=True, help='Matrix N size.')
    parser.add_argument('--K', type=int, required=True, help='Matrix K size.')
    parser.add_argument('--sparsity', type=int, required=True, help='Sparsity grade of the MK matrix.')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE, help='Tolerance for deviation on the results.')
    parser.add_argument('--generator', type=str, choices=['StonneMapper'], default='StonneMapper', help='Tile generator to use.')
    args = parser.parse_args()

    if args.sparsity < 0 or args.sparsity > 100:
        print('Invalid sparsity value. Must be between 0 and 100.')
        exit(1)

    if args.tolerance < 0 or args.tolerance > 1:
        print("Tolerance must be between 0 and 1.")
        exit(1)

    print('\nParameters used:')
    print(f'\tnum_ms: {args.num_ms}')
    print(f'\tdn_bw: {args.dn_bw}')
    print(f'\trn_bw: {args.rn_bw}')
    print(f'\tM: {args.M}')
    print(f'\tN: {args.N}')
    print(f'\tK: {args.K}')
    print(f'\tsparsity: {args.sparsity}')
    print(f'\ttolerance: {args.tolerance}')
    print(f'\tgenerator: {args.generator}')

    evaluate('ManualEvaluation', args.num_ms, args.dn_bw, args.rn_bw, args.M, args.N, args.K, args.sparsity, args.tolerance, args.generator)
