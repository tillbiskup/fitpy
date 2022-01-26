import os
import unittest

import numpy as np

import fitpy.dataset
import fitpy.report


class TestLaTeXFitReporter(unittest.TestCase):

    def setUp(self):
        self.reporter = fitpy.report.LaTeXFitReporter()
        self.dataset = fitpy.dataset.CalculatedDataset()
        self.filename = 'test_report.tex'
        self.result = 'test_report.pdf'
        self.figure_filename = 'test_report-fig.pdf'

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.result):
            os.remove(self.result)
        if os.path.exists(self.figure_filename):
            os.remove(self.figure_filename)

    def create_dataset(self):
        npoints = 100
        self.dataset.data.data = np.linspace(0, 1, npoints)
        self.dataset.data.residual = np.random.random(npoints) - 0.5

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

    def test_render_with_dataset_creates_figure(self):
        self.create_dataset()
        self.reporter.template = 'simplefit.tex'
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context['dataset'] = self.dataset.to_dict()
        self.reporter.render()
        self.assertTrue(os.path.exists(self.figure_filename))

    def test_render_with_dataset_adds_figure_filename_to_context(self):
        self.create_dataset()
        self.reporter.template = 'simplefit.tex'
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context['dataset'] = self.dataset.to_dict()
        self.reporter.render()
        # Note: Keys get converted from snake case to camel case for LaTeX!
        self.assertEqual(self.figure_filename,
                         self.reporter.context['figureFilename'])

    def test_render_with_dataset_adds_figure_filename_to_includes(self):
        self.create_dataset()
        self.reporter.template = 'simplefit.tex'
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context['dataset'] = self.dataset.to_dict()
        self.reporter.render()
        self.assertIn(self.figure_filename, self.reporter.includes)

    def test_create_with_dataset(self):
        self.create_dataset()
        self.reporter.template = 'simplefit.tex'
        self.reporter.dataset = self.dataset
        self.reporter.context['dataset'] = self.dataset.to_dict()
        self.reporter.filename = self.filename
        self.reporter.create()
        self.reporter.compile()
        self.assertTrue(os.path.exists(self.result))
