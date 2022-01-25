import unittest

import fitpy.report


class TestLaTeXFitReporter(unittest.TestCase):

    def setUp(self):
        reporter = fitpy.report.LaTeXFitReporter()

    def test_instantiate_class(self):
        pass
