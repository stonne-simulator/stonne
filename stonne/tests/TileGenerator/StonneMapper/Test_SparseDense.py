import os
import unittest
import subprocess
import random
try: # Try to import the local version first (usually works when executed from the command line with Python directly)
    import SparseDenseEvaluation as SparseDense
except ImportError: # Only works when you execute it with the '-m unittest' parameter from stonne/stonne directory
    import tests.TileGenerator.StonneMapper.SparseDenseEvaluation as SparseDense


# Available generators for SparseDense layers
GENERATOR = ["StonneMapper"]
GENERATOR = GENERATOR[0]

# Execution parameters
PERFORMANCE_TOLERANCE = 0.3
NUMBER_RANDOM_TESTS = 10


class TestSparseDense(unittest.TestCase):
    """
    Test cases to test the generation of StonneMapper for SparseDense layers.
    For each test, it runs a simulation of the layer generating the tile with StonneMapper.
    Later, it searches for the best possible tile for this layer.
    At last, it compares the generated tile results with the best possible tile results,
    passing the test only if the speedup fits in the tolerance margin.
    Note: accumulation_buffer is always 1
    """

    @classmethod
    def setUpClass(cls):
        # builds the STONNE executable
        proc = subprocess.Popen(["make", "all"])
        proc.wait()

    def test01_SparseDenseBasic1(self):
        results = []
        for sparsity in [10, 40, 70, 90]:
            results.append(SparseDense.evaluate(testname=f'Basic1-sp{sparsity}',
                num_ms=16, dn_bw=8, rn_bw=8, M=32, N=8, K=16, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))


    def test02_SparseDenseBasic2(self):
        results = []
        for sparsity in [1, 40, 70, 99]:
            results.append(SparseDense.evaluate(testname=f'Basic2-sp{sparsity}',
                num_ms=16, dn_bw=8, rn_bw=8, M=32, N=48, K=20, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test03_SparseDenseBasic3(self):
        results = []
        for sparsity in [1, 40, 70, 99]:
            results.append(SparseDense.evaluate(testname=f'Basic3-sp{sparsity}',
                num_ms=16, dn_bw=8, rn_bw=8, M=32, N=20, K=48, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test04_SparseDensePowersOfTwo1(self):
        results = []
        for sparsity in range(30, 70, 2):
            results.append(SparseDense.evaluate(testname=f'PowersOfTwo1-sp{sparsity}',
                num_ms=32, dn_bw=32, rn_bw=32, M=8, N=64, K=32, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test05_SparseDensePowersOfTwo2(self):
        results = []
        for sparsity in range(1, 100, 5):
            results.append(SparseDense.evaluate(testname=f'PowersOfTwo2-sp{sparsity}',
                num_ms=64, dn_bw=64, rn_bw=64, M=32, N=128, K=32, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test06_SparseDensePowersOfTwo3(self):
        results = []
        for sparsity in range(1, 100, 5):
            results.append(SparseDense.evaluate(testname=f'PowersOfTwo3-sp{sparsity}',
                num_ms=128, dn_bw=128, rn_bw=64, M=16, N=64, K=128, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test07_SparseDensePowersOfTwo4(self):
        results = []
        for sparsity in range(1, 100, 7):
            results.append(SparseDense.evaluate(testname=f'PowersOfTwo4-sp{sparsity}',
                num_ms=256, dn_bw=256, rn_bw=64, M=16, N=256, K=256, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test08_SparseDensePowersOfTwo5(self):
        results = []
        for sparsity in range(1, 100, 7):
            results.append(SparseDense.evaluate(testname=f'PowersOfTwo5-sp{sparsity}',
                num_ms=512, dn_bw=512, rn_bw=128, M=4, N=128, K=256, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test09_SparseDensePowersOfTwo6(self):
        results = []
        for sparsity in range(0, 100, 10):
            results.append(SparseDense.evaluate(testname=f'PowersOfTwo6-sp{sparsity}',
                num_ms=512, dn_bw=256, rn_bw=128, M=4, N=1024, K=64, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test10_SparseDenseRandom1(self):
        results = []
        for sparsity in range(1, 100, 5):
            results.append(SparseDense.evaluate(testname=f'Random1-sp{sparsity}',
                num_ms=8, dn_bw=8, rn_bw=8, M=7, N=7, K=9, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test11_SparseDenseRandom2(self):
        results = []
        for sparsity in range(1, 100, 5):
            results.append(SparseDense.evaluate(testname=f'Random2-sp{sparsity}',
                num_ms=16, dn_bw=16, rn_bw=16, M=5, N=17, K=9, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test12_SparseDenseRandom3(self):
        results = []
        for sparsity in range(1, 100, 5):
            results.append(SparseDense.evaluate(testname=f'Random3-sp{sparsity}',
                num_ms=32, dn_bw=32, rn_bw=16, M=5, N=31, K=17, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test13_SparseDenseRandom4(self):
        results = []
        for sparsity in range(1, 100, 5):
            results.append(SparseDense.evaluate(testname=f'Random4-sp{sparsity}',
                num_ms=64, dn_bw=32, rn_bw=16, M=3, N=70, K=91, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test14_SparseDenseGenerateRandomly(self):
        results = []
        for i in range(NUMBER_RANDOM_TESTS):
            num_ms = 2 ** random.randint(3, 9) # 8..512
            dn_bw = num_ms
            rn_bw = num_ms
            M = random.randint(1, 256)
            N = random.randint(1, 256)
            K = random.randint(1, 256)
            sparsity = random.randint(0, 99)
            results.append(SparseDense.evaluate(testname=f'RandomGenerated[{i}]',
                num_ms=num_ms, dn_bw=dn_bw, rn_bw=rn_bw, M=M, N=N, K=K, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))


# Main method to execute all testcases of StonneMapper for SparseDense layers
if __name__ == "__main__":
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this test script from the stonne/stonne directory")
        exit(1)

    unittest.main()
