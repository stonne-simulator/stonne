import os
import unittest
import subprocess
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


# Main method to execute all testcases of MyGenerator for SparseDense layers
if __name__ == "__main__":
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this test script from the stonne/stonne directory")
        exit(1)

    unittest.main()
