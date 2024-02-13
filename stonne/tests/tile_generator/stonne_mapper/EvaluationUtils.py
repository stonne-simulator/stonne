import re
import csv
import datetime

# CSV output filenames with timestamp (same file during all tests)
DENSEGEMM_CSV_FILENAME = f"DenseGemmEvaluation-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
SPARSEDENSE_CSV_FILENAME = f"SparseDenseEvaluation-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"


def get_cycles(stdout):
    match = re.search(r'Number of cycles running: (\d+)', stdout)
    if match:
        return int(match.group(1))
    else:
        return 0


def get_densegemm_tile(stdout):
    match = re.search(r'Generated tile: <T_M=(\d+), T_N=(\d+), T_K=(\d+)>', stdout)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    else:
        return None


def get_sparsedense_tile(stdout):
    match = re.search(r'Generated tile: <T_N=(\d+), T_K=(\d+)>', stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None


def check_assertions(stderr):
    match = re.search(r'Assertion', stderr)
    return True if match else False


def save_densegemm_results_csv(passed, testname, num_ms, dn_bw, rn_bw, M, N, K, generator, generatedtile, generatedtile_cycles, min_tile, min_cycles, speedup, tolerance):
    fields = ['PASSED', 'TESTNAME', 'NUM_MS', 'DN_BW', 'RN_BW', 'M', 'N', 'K', 'GENERATOR', 'GEN-T_M', 'GEN-T_N', 'GEN-T_K',
              'GEN-CYCLES', 'MIN-T_M', 'MIN-T_N', 'MIN-T_K', 'MIN-CYCLES', 'SPEEDUP', 'TOLERANCE']
    with open(DENSEGEMM_CSV_FILENAME, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # if the file is empty, write the header
        if not csvfile.tell():
            writer.writeheader()

        # write the row with the results of the simulation to the csv file
        writer.writerow({'PASSED': passed,
                         'TESTNAME': testname,
                         'NUM_MS': num_ms,
                         'DN_BW': dn_bw,
                         'RN_BW': rn_bw,
                         'M': M,
                         'N': N,
                         'K': K,
                         'GENERATOR': generator,
                         'GEN-T_M': generatedtile[0],
                         'GEN-T_N': generatedtile[1],
                         'GEN-T_K': generatedtile[2],
                         'GEN-CYCLES': generatedtile_cycles,
                         'MIN-T_M': min_tile[0],
                         'MIN-T_N': min_tile[1],
                         'MIN-T_K': min_tile[2],
                         'MIN-CYCLES': min_cycles,
                         'SPEEDUP': speedup,
                         'TOLERANCE': tolerance})


def save_sparsedense_results_csv(passed, testname, num_ms, dn_bw, rn_bw, M, N, K, sparsity, generator, generatedtile, generatedtile_cycles, min_tile, min_cycles, speedup, tolerance):
    fields = ['PASSED', 'TESTNAME', 'NUM_MS', 'DN_BW', 'RN_BW',  'M', 'N', 'K', 'SPARSITY', 'GENERATOR', 'GEN-T_N', 'GEN-T_K',
              'GEN-CYCLES', 'MIN-T_N', 'MIN-T_K', 'MIN-CYCLES', 'SPEEDUP', 'TOLERANCE']
    with open(SPARSEDENSE_CSV_FILENAME, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # if the file is empty, write the header
        if not csvfile.tell():
            writer.writeheader()

        # write the row with the results of the simulation to the csv file
        writer.writerow({'PASSED': passed,
                         'TESTNAME': testname,
                         'NUM_MS': num_ms,
                         'DN_BW': dn_bw,
                         'RN_BW': rn_bw,
                         'M': M,
                         'N': N,
                         'K': K,
                         'SPARSITY': sparsity,
                         'GENERATOR': generator,
                         'GEN-T_N': generatedtile[0],
                         'GEN-T_K': generatedtile[1],
                         'GEN-CYCLES': generatedtile_cycles,
                         'MIN-T_N': min_tile[0],
                         'MIN-T_K': min_tile[1],
                         'MIN-CYCLES': min_cycles,
                         'SPEEDUP': speedup,
                         'TOLERANCE': tolerance})
