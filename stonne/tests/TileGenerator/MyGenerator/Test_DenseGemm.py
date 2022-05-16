import os
import unittest
import subprocess
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

    def testDenseGemmBasic1(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=2
        # - Total Cycles for best tile: 560
        self.assertTrue(DenseGemm.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=16, N=16, K=16, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmBasic2(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=2
        # - Total Cycles for best tile: 536
        self.assertTrue(DenseGemm.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=32, N=8, K=16, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmBasic3(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=2
        # - Total Cycles for best tile: 72
        self.assertTrue(DenseGemm.evaluate(num_ms=16, dn_bw=8, rn_bw=8, M=7, N=7, K=7, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmPowerOfTwo1(self):
        # - Best tile (using powers of 2): T_M=1, T_N=128, T_K=2
        # - Total Cycles for best tile: 4150
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=256, M=128, N=256, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmPowerOfTwo2(self):
        # - Best tile (using powers of 2): T_M=1, T_N=8, T_K=16
        # - Total Cycles for best tile: 1042
        self.assertTrue(DenseGemm.evaluate(num_ms=128, dn_bw=128, rn_bw=64, M=128, N=8, K=128, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmPowerOfTwo3(self):
        # - Best tile (using powers of 2): T_M=1, T_N=32, T_K=2
        # - Total Cycles for best tile: 2194
        self.assertTrue(DenseGemm.evaluate(num_ms=64, dn_bw=64, rn_bw=64, M=16, N=64, K=128, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmRandom1(self):
        # - Best tile (using powers of 2): T_M=1, T_N=64, T_K=4
        # - Total Cycles for best tile: 4735
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=64, M=116, N=317, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmRandom2(self):
        # - Best tile (using powers of 2): T_M=1, T_N=32, T_K=8
        # - Total Cycles for best tile: 5269
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=64, M=116, N=347, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    def testDenseGemmRandom3(self):
        # - Best tile (using powers of 2): T_M=1, T_N=16, T_K=16
        # - Total Cycles for best tile: 5145
        self.assertTrue(DenseGemm.evaluate(num_ms=256, dn_bw=256, rn_bw=64, M=116, N=333, K=32, tolerance=PERFORMANCE_TOLERANCE, generator=GENERATOR))

    # TODO: implement more tests

# Main method to execute all testcases of MyGenerator for DenseGEMM/FC layers
if __name__ == "__main__":
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this DenseGemm.evaluate script from the stonne/stonne directory")
        exit(1)

    unittest.main()
