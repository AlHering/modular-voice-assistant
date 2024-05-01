# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import unittest
from src.quality.configuration_tests import test_configuration
from src.quality.model_tests import test_dataclasses
from src.quality.model_tests import test_llm_pool
from src.quality.control_tests import test_backend_controller

loader = unittest.TestLoader()
suite = unittest.TestSuite()
suite.addTests(loader.loadTestsFromModule(test_configuration))
suite.addTests(loader.loadTestsFromModule(test_dataclasses))
suite.addTests(loader.loadTestsFromModule(test_llm_pool))
suite.addTests(loader.loadTestsFromModule(test_backend_controller))


if __name__ == "__main__":
    # TODO: Investigate problems with multithreaded llm handling in backend controller tests.
    unittest.TextTestRunner(verbosity=3).run(suite)
