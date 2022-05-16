import unittest
import os
try: # Try to import the local version first (usually works when executed from the command line with Python directly)
    import MyGenerator.Test_DenseGemm as Test_DenseGemm
    #import MyGenerator.Test_SparseDense as Test_SparseDense
except ImportError: # Only works when you execute it with the '-m unittest' parameter from stonne/stonne directory
    import tests.TileGenerator.MyGenerator.Test_DenseGemm as Test_DenseGemm
    #import tests.TileGenerator.MyGenerator.Test_SparseDense as Test_SparseDense


if __name__ == "__main__":
    """
    Executes the all the acceptance tests cases for MyGenerator module.
    """
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this test script from the stonne/stonne directory")
        exit(1)

    # add all testcases to suite case
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test_DenseGemm.TestDenseGemm))
    #suite.addTest(unittest.makeSuite(Test_SparseDense.TestSparseDense))

    # run tests
    unittest.TextTestRunner(verbosity=2).run(suite)
