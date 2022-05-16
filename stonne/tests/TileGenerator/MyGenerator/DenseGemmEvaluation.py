import argparse
from math import log2
import os
import subprocess
import sys
try: # Try to import the local version first (usually works when executed from the command line with Python directly)
    import EvaluationUtils
except ImportError: # Only works when you execute it with the '-m unittest' parameter from stonne/stonne directory
    import tests.TileGenerator.MyGenerator.EvaluationUtils as EvaluationUtils


DEFAULT_TOLERANCE = 0.1


def evaluate(num_ms, dn_bw, rn_bw, M, N, K, tolerance=DEFAULT_TOLERANCE, generator="MyGenerator"):
    ### Results from the generator ###

    # execution generating a new tile automatically
    process = subprocess.Popen(['./stonne', '-FC', f'-M={M}', f'-N={N}', f'-K={K}',
                                f'-num_ms={num_ms}', f'-dn_bw={dn_bw}', f'-rn_bw={rn_bw}', '-accumulation_buffer=1',
                                '-generate_tile=performance', f'-generator={generator}'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    # get results from execution
    generatedtile_cycles = EvaluationUtils.get_cycles(stdout)
    if generatedtile_cycles == 0 or EvaluationUtils.check_assertions(stderr):
        print('There was an error during execution. Cause:', file=sys.stderr)
        print(stderr, file=sys.stderr)
        return False

    generatedtile = EvaluationUtils.get_densegemm_tile(stdout)


    ### Search of the best mapping using powers of 2 ###

    # search for best mapping trying all combinations
    min_cycles = 9999999999
    min_tile = None
    print('\n# Trying all combinations searching the best tile using powers of 2')
    for T_M in [2 ** i for i in range(0, int(log2(num_ms)) + 1)]:
        for T_N in [2 ** i for i in range(0, int(log2(num_ms)) + 1)]:
            for T_K in [2 ** i for i in range(0, int(log2(num_ms)) + 1)]:
                # ensure that num_ms occupation is maximum and tile size does not exceed its dimension,
                # although allowing to slightly exceed the limit
                if T_M >= M * 2 or T_N >= N * 2 or T_K >= K * 2 or T_K == 1 or T_M * T_N * T_K != num_ms:
                    continue

                # execution using a fixed tile
                process = subprocess.Popen(['./stonne', '-FC', f'-M={M}', f'-N={N}', f'-K={K}',
                                            f'-num_ms={num_ms}', f'-dn_bw={dn_bw}', f'-rn_bw={rn_bw}',
                                            '-accumulation_buffer=1', f'-T_M={T_M}', f'-T_N={T_N}', f'-T_K={T_K}'],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                stdout = stdout.decode('utf-8')
                stderr = stderr.decode('utf-8')

                # get results from execution
                cycles = EvaluationUtils.get_cycles(stdout)
                if generatedtile_cycles == 0 or EvaluationUtils.check_assertions(stderr):
                    # if this execution failed, try the next combination)
                    continue

                # updates the best tile found
                if cycles < min_cycles:
                    min_cycles = cycles
                    min_tile = [T_M, T_N, T_K]

                print(f'\t- T_M={T_M}, T_N={T_N}, T_K={T_K} => cycles={cycles}')

    # get and print final results
    speedup = ((1 / generatedtile_cycles) / (1 / min_cycles))
    passed = True if 1 - speedup <= tolerance else False
    print()
    print('# Final results')
    print(f' - Matrix sizes: M={M}, N={N}, K={K}')
    print(f' - Tile Generated Automatically: T_M={generatedtile[0]}, T_N={generatedtile[1]}, T_K={generatedtile[2]}')
    print(f' - Total Cycles for generated tile: {generatedtile_cycles}')
    print(f' - Best tile found: T_M={min_tile[0]}, T_N={min_tile[1]}, T_K={min_tile[2]}')
    print(f' - Total Cycles for best tile: {min_cycles}')
    print(f' - Speedup of the generated tile: {speedup}')
    print(f' - Pass the test (tolerance={tolerance}) ? => {passed}')

    EvaluationUtils.save_densegemm_results_csv(passed, M, N, K, generator, generatedtile, generatedtile_cycles, min_tile, min_cycles, speedup, tolerance)

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
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE, help='Tolerance for deviation on the results.')
    parser.add_argument('--generator', type=str, choices=['mRNA', 'MyGenerator'], default='MyGenerator', help='Tile generator to use.')
    args = parser.parse_args()

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
    print(f'\ttolerance: {args.tolerance}')
    print(f'\tgenerator: {args.generator}')

    evaluate(args.num_ms, args.dn_bw, args.rn_bw, args.M, args.N, args.K, args.tolerance, args.generator)
