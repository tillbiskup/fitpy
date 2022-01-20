import unittest

import matplotlib.pyplot as plt
import numpy as np

import aspecd.dataset
import aspecd.model
import aspecd.plotting

import fitpy.analysis


class TestSimpleFit(unittest.TestCase):
    def setUp(self):
        self.fit = fitpy.analysis.SimpleFit()
        self.model = aspecd.model.Gaussian()

    def create_dataset(self):
        model = aspecd.model.Gaussian()
        model.variables = [np.linspace(-10, 10, 1001)]
        model.parameters['position'] = 2
        self.dataset = model.create()

    def plot_dataset(self, dataset):
        plotter = aspecd.plotting.SinglePlotter1D()
        dataset.plot(plotter)
        plt.show()

    def test_instantiate_class(self):
        pass

    def test_has_model_property(self):
        self.assertTrue(hasattr(self.fit, 'model'))

    def test_has_sensible_description(self):
        self.assertIn('Fit model to data of dataset',
                      self.fit.description)

    def test_has_fit_parameter(self):
        self.assertTrue('fit' in self.fit.parameters)

    def test_analysis_returns_calculated_dataset(self):
        self.create_dataset()
        self.fit.model = self.model
        fit = self.dataset.analyse(self.fit)
        self.assertIsInstance(fit.result,
                              aspecd.dataset.CalculatedDataset)

    def test_analyse_returns_fitted_data(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_returned_dataset_has_same_x_axis_as_model(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertListEqual(list(self.dataset.data.axes[0].values),
                             list(fit.result.data.axes[0].values))

    def test_analyse_with_parameter_range(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0,
                                                   'range': [-1, 1]}}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(1., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_analyse_with_multiple_parameters_and_range(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {
            'position': {'start': 0, 'range': [-1, 3]},
            'amplitude': {'start': 1.5, 'range': [0.5, 3]},
        }
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])
        self.assertAlmostEqual(1., np.max(fit.result.data.data), 8)

    def test_returned_dataset_contains_residual(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertListEqual(list(np.zeros(len(self.dataset.data.data))),
                             list(fit.result.data.residual))
