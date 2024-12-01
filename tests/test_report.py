import contextlib
import io
import os
import unittest

import aspecd.model
import aspecd.processing
import numpy as np

import fitpy.analysis
import fitpy.dataset
import fitpy.report


class TestLaTeXFitReporter(unittest.TestCase):
    def setUp(self):
        self.reporter = fitpy.report.LaTeXFitReporter()
        self.dataset = fitpy.dataset.CalculatedDataset()
        self.filename = "test_report.tex"
        self.result = "test_report.pdf"
        self.figure_filename = "test_report-fig.pdf"

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.result):
            os.remove(self.result)
        if os.path.exists(self.figure_filename):
            os.remove(self.figure_filename)

    def create_dataset(self):
        model = aspecd.model.Gaussian()
        model.variables = [np.linspace(-10, 10, 1001)]
        model.parameters["position"] = 2
        data = model.create()
        noise = aspecd.processing.Noise()
        noise.parameters["amplitude"] = 0.2
        data.process(noise)
        data.id = "foobar"
        data.label = "Some random spectral line"

        fit = fitpy.analysis.SimpleFit()
        fit.model = model
        fit.parameters["fit"] = {"position": {"start": 0}}
        fit = data.analyse(fit)

        self.dataset = fit.result

    def test_instantiate_class(self):
        pass

    def test_render(self):
        self.create_dataset()
        self.reporter.template = "simplefit.tex"
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        self.assertTrue(self.reporter.report)

    def test_create(self):
        self.create_dataset()
        self.reporter.template = "simplefit.tex"
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.filename = self.filename
        self.reporter.create()
        with contextlib.redirect_stdout(io.StringIO()):
            self.reporter.compile()
        self.assertTrue(os.path.exists(self.result))

    def test_render_with_dataset_creates_figure(self):
        self.create_dataset()
        self.reporter.template = "simplefit.tex"
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        self.assertTrue(os.path.exists(self.figure_filename))

    def test_render_with_dataset_adds_figure_filename_to_context(self):
        self.create_dataset()
        self.reporter.template = "simplefit.tex"
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        # Note: Keys get converted from snake case to camel case for LaTeX!
        self.assertEqual(
            self.figure_filename, self.reporter.context["figureFilename"]
        )

    def test_render_with_dataset_adds_figure_filename_to_includes(self):
        self.create_dataset()
        self.reporter.template = "simplefit.tex"
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        self.assertIn(self.figure_filename, self.reporter.includes)

    def test_create_with_dataset(self):
        self.create_dataset()
        self.reporter.template = "simplefit.tex"
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.filename = self.filename
        self.reporter.create()
        with contextlib.redirect_stdout(io.StringIO()):
            self.reporter.compile()
        self.assertTrue(os.path.exists(self.result))


class TestLaTeXLHSFitReporter(unittest.TestCase):
    def setUp(self):
        self.reporter = fitpy.report.LaTeXLHSFitReporter()
        self.dataset = fitpy.dataset.CalculatedDataset()
        self.filename = "test_report.tex"
        self.result = "test_report.pdf"
        self.figure_filename = "test_report-fig.pdf"
        self.lhs_figure_filename = "test_report-lhsfig.pdf"

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
        if os.path.exists(self.result):
            os.remove(self.result)
        if os.path.exists(self.figure_filename):
            os.remove(self.figure_filename)
        if os.path.exists(self.lhs_figure_filename):
            os.remove(self.lhs_figure_filename)

    def create_dataset(self):
        model = aspecd.model.Gaussian()
        model.variables = [np.linspace(-10, 10, 1001)]
        model.parameters["position"] = 2
        data = model.create()
        noise = aspecd.processing.Noise()
        noise.parameters["amplitude"] = 0.2
        data.process(noise)
        data.id = "foobar"
        data.label = "Some random spectral line"

        fit = fitpy.analysis.LHSFit()
        fit.model = model
        fit.parameters["fit"] = {"position": {"lhs_range": [-8, 8]}}
        fit.parameters["lhs"] = {"points": 5}
        fit = data.analyse(fit)

        self.dataset = fit.result

    def test_instantiate_class(self):
        pass

    def test_render(self):
        self.create_dataset()
        self.reporter.template = "lhsfit.tex"
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        self.assertTrue(self.reporter.report)

    def test_create(self):
        self.create_dataset()
        self.reporter.template = "lhsfit.tex"
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.filename = self.filename
        self.reporter.create()
        with contextlib.redirect_stdout(io.StringIO()):
            self.reporter.compile()
        self.assertTrue(os.path.exists(self.result))

    def test_render_with_dataset_creates_lhs_figure(self):
        self.create_dataset()
        self.reporter.template = "lhsfit.tex"
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        self.assertTrue(os.path.exists(self.lhs_figure_filename))

    def test_render_with_dataset_adds_lhs_figure_filename_to_context(self):
        self.create_dataset()
        self.reporter.template = "lhsfit.tex"
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        # Note: Keys get converted from snake case to camel case for LaTeX!
        self.assertEqual(
            self.lhs_figure_filename,
            self.reporter.context["lhsFigureFilename"],
        )

    def test_render_with_dataset_adds_lhs_figure_filename_to_includes(self):
        self.create_dataset()
        self.reporter.template = "lhsfit.tex"
        self.reporter.filename = self.filename
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.render()
        self.assertIn(self.lhs_figure_filename, self.reporter.includes)

    def test_create_with_dataset(self):
        self.create_dataset()
        self.reporter.template = "lhsfit.tex"
        self.reporter.dataset = self.dataset
        self.reporter.context["dataset"] = self.dataset.to_dict()
        self.reporter.filename = self.filename
        self.reporter.create()
        with contextlib.redirect_stdout(io.StringIO()):
            self.reporter.compile()
        self.assertTrue(os.path.exists(self.result))
