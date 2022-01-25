import os
import unittest

import fitpy.dataset
import fitpy.report


class TestLaTeXFitReporter(unittest.TestCase):

    def setUp(self):
        self.reporter = fitpy.report.LaTeXFitReporter()
        self.dataset = fitpy.dataset.CalculatedDataset()
        self.filename = 'test_report.tex'
        self.result = 'test_report.pdf'

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.result):
            os.remove(self.result)

    def test_instantiate_class(self):
        pass

    def test_render(self):
        self.reporter.template = 'simplefit.tex'
        self.reporter.context['dataset'] = self.dataset.to_dict()
        self.reporter.render()
        self.assertTrue(self.reporter.report)

    def test_create(self):
        self.reporter.template = 'simplefit.tex'
        self.reporter.context['dataset'] = self.dataset.to_dict()
        self.reporter.filename = self.filename
        self.reporter.create()
        self.reporter.compile()
        self.assertTrue(os.path.exists(self.result))
