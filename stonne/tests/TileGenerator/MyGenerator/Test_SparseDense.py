import os
import unittest
import subprocess
import random
try: # Try to import the local version first (usually works when executed from the command line with Python directly)
    import SparseDenseEvaluation as SparseDense
except ImportError: # Only works when you execute it with the '-m unittest' parameter from stonne/stonne directory
    import tests.TileGenerator.MyGenerator.SparseDenseEvaluation as SparseDense


PERFORMANCE_TOLERANCE = 0.2
GENERATOR = "MyGenerator"


class TestSparseDense(unittest.TestCase):
    """
    Test cases to test the generation of MyGenerator for SparseDense layers.
    For each test, it runs a simulation of the layer generating the tile with MyGenerator.
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

    def testSparseDenseBasic1(self):
        for sparsity in [10, 40, 70, 90]:
            self.assertTrue(SparseDense.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=32, N=8, K=16, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseBasic2(self):
        for sparsity in [1, 40, 70, 99]:
            self.assertTrue(SparseDense.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=32, N=48, K=20, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseBasic3(self):
        for sparsity in [1, 40, 70, 99]:
            self.assertTrue(SparseDense.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=32, N=20, K=48, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDensePowersOfTwo1(self):
        for sparsity in range(30, 70, 2):
            self.assertTrue(SparseDense.evaluate(num_ms=32, dn_bw=32, rn_bw=32, M=8, N=64, K=32, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDensePowersOfTwo2(self):
        for sparsity in range(1, 100, 5):
            self.assertTrue(SparseDense.evaluate(num_ms=64, dn_bw=64, rn_bw=64, M=32, N=128, K=32, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDensePowersOfTwo3(self):
        for sparsity in range(1, 100, 5):
            self.assertTrue(SparseDense.evaluate(num_ms=128, dn_bw=128, rn_bw=64, M=16, N=64, K=128, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDensePowersOfTwo4(self):
        for sparsity in range(1, 100, 7):
            self.assertTrue(SparseDense.evaluate(num_ms=256, dn_bw=256, rn_bw=64, M=16, N=256, K=256, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDensePowersOfTwo5(self):
        for sparsity in range(1, 100, 7):
            self.assertTrue(SparseDense.evaluate(num_ms=512, dn_bw=512, rn_bw=128, M=4, N=128, K=256, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDensePowersOfTwo6(self):
        for sparsity in range(0, 100, 10):
            self.assertTrue(SparseDense.evaluate(num_ms=512, dn_bw=256, rn_bw=128, M=4, N=1024, K=64, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseRandom1(self):
        for sparsity in range(1, 100, 5):
            self.assertTrue(SparseDense.evaluate(num_ms=8, dn_bw=8, rn_bw=8, M=7, N=7, K=9, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseRandom2(self):
        for sparsity in range(1, 100, 5):
            self.assertTrue(SparseDense.evaluate(num_ms=16, dn_bw=16, rn_bw=16, M=5, N=17, K=9, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseRandom3(self):
        for sparsity in range(1, 100, 5):
            self.assertTrue(SparseDense.evaluate(num_ms=32, dn_bw=32, rn_bw=16, M=5, N=31, K=17, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseRandom4(self):
        for sparsity in range(1, 100, 5):
            self.assertTrue(SparseDense.evaluate(num_ms=64, dn_bw=32, rn_bw=16, M=3, N=70, K=91, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseGenerateRandomly(self):
        for i in range(100):
            max_power = random.randint(1, 10) # 2..1024
            num_ms = 2 ** max_power
            dn_bw = 2 ** random.randint(0, max_power) # 1..num_ms
            rn_bw = 2 ** random.randint(0, max_power) # 1..num_ms
            M = random.randint(1, 1024)
            N = random.randint(1, 1024)
            K = random.randint(1, 1024)
            sparsity = random.randint(0, 99)
            self.assertTrue(SparseDense.evaluate(num_ms=num_ms, dn_bw=dn_bw, rn_bw=rn_bw, M=M, N=N, K=K, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseGenerateRandomlyFixHardware(self):
        for i in range(100):
            max_power = random.randint(1, 10) # 2..1024
            num_ms = 2 ** max_power
            dn_bw = num_ms
            rn_bw = num_ms
            M = random.randint(1, 1024)
            N = random.randint(1, 1024)
            K = random.randint(1, 1024)
            sparsity = random.randint(0, 99)
            self.assertTrue(SparseDense.evaluate(num_ms=num_ms, dn_bw=dn_bw, rn_bw=rn_bw, M=M, N=N, K=K, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testSparseDenseGenerateRandomlyFixHardwareAlwaysLess(self):
        for i in range(100):
            max_power = random.randint(1, 10) # 2..1024
            num_ms = 2 ** max_power
            dn_bw = num_ms
            rn_bw = num_ms
            M = random.randint(1, num_ms)
            N = random.randint(1, num_ms)
            K = random.randint(1, num_ms)
            sparsity = random.randint(0, 99)
            self.assertTrue(SparseDense.evaluate(num_ms=num_ms, dn_bw=dn_bw, rn_bw=rn_bw, M=M, N=N, K=K, sparsity=sparsity, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))


# Main method to execute all testcases of MyGenerator for SparseDense layers
if __name__ == "__main__":
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this test script from the stonne/stonne directory")
        exit(1)

    unittest.main()
