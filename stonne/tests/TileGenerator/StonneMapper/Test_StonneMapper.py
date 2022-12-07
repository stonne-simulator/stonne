import unittest
import os
# since it not include an unittest.TestCase, it has to be run directly with python Test_StonneMapper.py
import Test_DenseGemm
import Test_DenseGemmRealModels
import Test_SparseDense
import Test_SparseDenseRealModels


if __name__ == "__main__":
    """
    Executes the all the acceptance tests cases for StonneMapper module.
    """
    if not os.getcwd().endswith('stonne/stonne'):
        print("Please run this test script from the stonne/stonne directory")
        exit(1)

    # add all testcases to suite case
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(Test_DenseGemm.TestDenseGemm))
    suite.addTest(unittest.makeSuite(Test_DenseGemmRealModels.TestDenseGemmRealModels))
    suite.addTest(unittest.makeSuite(Test_SparseDense.TestSparseDense))
    suite.addTest(unittest.makeSuite(Test_SparseDenseRealModels.TestSparseDenseRealModels))

    # run tests
    unittest.TextTestRunner(verbosity=2).run(suite)
