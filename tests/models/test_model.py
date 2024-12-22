import unittest
from tests.models.test_evaluate import TestEvaluateModel
from tests.models.test_predict import TestPredictModel
from tests.models.test_train import TestTrainModel


def suite():
    """Combine all tests into a single test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestEvaluateModel))
    suite.addTest(unittest.makeSuite(TestPredictModel))
    suite.addTest(unittest.makeSuite(TestTrainModel))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
