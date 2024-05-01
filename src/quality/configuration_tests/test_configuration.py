# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import unittest
import re
import requests
from src.configuration import configuration as cfg


class ConfigurationTest(unittest.TestCase):
    """
    Test case class for testing the configuration base.
    """

    def test_urls(self) -> None:
        """
        Test method for testing configured URLs.
        """
        for attribute in [attribute for attribute in dir(cfg.URLS) if re.fullmatch(r"^[A-Z]+?_URL", attribute)]:
            resp = requests.get(getattr(cfg.URLS, attribute))
            self.assertEqual(resp.status_code, 200)

    def test_paths(self) -> None:
        """
        Test method for testing configured paths.
        """
        for attribute in [attribute.replace(" =", "") for attribute in dir(cfg.PATHS) if re.fullmatch(r"^[A-Z]+?_PATH =", attribute)]:
            path = getattr(cfg.PATHS, attribute)
            self.assertTrue(re.fullmatch(
                r"^(/|[A-Z]:\\){1}((/|\\)[a-zA-Z0-9#~\-_,'!\"§\$%&\(\)\[\]=`'])+", path))
            self.assertTrue(cfg.PATHS.PACKAGE_PATH in path)

    def test_configuration(self) -> None:
        """
        Test method for testing central configuration.
        """
        for attribute in ["ENV", "LOGGER", "BACKEND_HOST", "BACKEND_PORT"]:
            self.assertTrue(hasattr(cfg, attribute))
            self.assertFalse(getattr(cfg, attribute) is None)

    @classmethod
    def setUpClass(cls):
        """
        Class method for setting up test case.
        """
        pass

    @classmethod
    def tearDownClass(cls):
        """
        Class method for setting tearing down test case.
        """
        pass

    @classmethod
    def setup_class(cls):
        """
        Alternative class method for setting up test case.
        """
        cls.setUpClass()

    @classmethod
    def teardown_class(cls):
        """
        Alternative class for setting tearing down test case.
        """
        cls.tearDownClass()


if __name__ == '__main__':
    unittest.main()
