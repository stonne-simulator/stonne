import os
import unittest
import subprocess
import random
try: # Try to import the local version first (usually works when executed from the command line with Python directly)
    import DenseGemmEvaluation as DenseGemm
except ImportError: # Only works when you execute it with the '-m unittest' parameter from stonne/stonne directory
    import tests.TileGenerator.MyGenerator.DenseGemmEvaluation as DenseGemm


PERFORMANCE_TOLERANCE = 0.2
GENERATOR = "MyGenerator"


class TestDenseGemm(unittest.TestCase):
    """
    Test cases to test the generation of MyGenerator for DenseGemm layers.
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

    def test01_DenseGemmBasic1(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=2
        # - Total Cycles for best tile: 560
        self.assertTrue(DenseGemm.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=16, N=16, K=16, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test02_DenseGemmBasic2(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=2
        # - Total Cycles for best tile: 536
        self.assertTrue(DenseGemm.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=32, N=8, K=16, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test03_DenseGemmBasic3(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=2
        # - Total Cycles for best tile: 72
        self.assertTrue(DenseGemm.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=7, N=7, K=7, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test04_DenseGemmPowerOfTwo1(self):
        # - Best tile (using powers of 2): T_M=1, T_N=128, T_K=2
        # - Total Cycles for best tile: 4150
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=256, M=128, N=256, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test05_DenseGemmPowerOfTwo2(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=16
        # - Total Cycles for best tile: 1042
        self.assertTrue(DenseGemm.evaluate(num_ms=128, dn_bw=128, rn_bw=64, M=128, N=8, K=128, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test06_DenseGemmPowerOfTwo3(self):
        # - Best tile (using powers of 2): T_M=1, T_N=32, T_K=2
        # - Total Cycles for best tile: 2194
        self.assertTrue(DenseGemm.evaluate(num_ms=64, dn_bw=64, rn_bw=64, M=16, N=64, K=128, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test07_DenseGemmRandom1(self):
        # - Best tile (using powers of 2): T_M=1, T_N=64, T_K=4
        # - Total Cycles for best tile: 4735
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=64, M=116, N=317, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test08_DenseGemmRandom2(self):
        # - Best tile (using powers of 2): T_M=1, T_N=32, T_K=8
        # - Total Cycles for best tile: 5269
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=64, M=116, N=347, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test09_DenseGemmRandom3(self):
        # - Best tile (using powers of 2): T_M=1, T_N=16, T_K=16
        # - Total Cycles for best tile: 5145
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=64, M=116, N=333, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def test10_DenseGemmGenerateRandomly(self):
        results = []
        for i in range(10):
            num_ms = 2 ** random.randint(3, 10) # 8..1024
            dn_bw = num_ms
            rn_bw = num_ms
            M = random.randint(1, 256)
            N = random.randint(1, 256)
            K = random.randint(1, 256)
            results.append(DenseGemm.evaluate(num_ms=num_ms, dn_bw=dn_bw, rn_bw=rn_bw, M=M, N=N, K=K, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))

    def test11_DenseGemmGenerateRandomlyAlwaysLess(self):
        results = []
        for i in range(10):
            max_power = random.randint(3, 10) # 8..1024
            num_ms = 2 ** max_power
            dn_bw = num_ms
            rn_bw = num_ms
            M = random.randint(1, num_ms)
            N = random.randint(1, num_ms)
            K = random.randint(1, num_ms)
            results.append(DenseGemm.evaluate(num_ms=num_ms, dn_bw=dn_bw, rn_bw=rn_bw, M=M, N=N, K=K, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))
        self.assertTrue(all(results))


# Main method to execute all testcases of MyGenerator for DenseGEMM/FC layers
if __name__ == "__main__":
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this DenseGemm.evaluate script from the stonne/stonne directory")
        exit(1)

    unittest.main()
