import unittest

import matplotlib.pyplot as plt
import numpy as np

import aspecd.dataset
import aspecd.model
import aspecd.plotting
import aspecd.utils

import fitpy.analysis
import fitpy.dataset


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

    def test_has_algorithm_parameter(self):
        self.assertTrue('algorithm' in self.fit.parameters)

    def test_analysis_returns_calculated_dataset(self):
        self.create_dataset()
        self.fit.model = self.model
        fit = self.dataset.analyse(self.fit)
        self.assertIsInstance(fit.result,
                              fitpy.dataset.CalculatedDataset)

    def test_analyse_returns_fitted_data(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_analyse_sets_algorithm_description(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertTrue(fit.parameters['algorithm']['description'])

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

    def test_returned_dataset_contains_model_metadata(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(aspecd.utils.full_class_name(self.model),
                         fit.result.metadata.model.type)
        self.assertDictEqual(fit.model.parameters,
                             fit.result.metadata.model.parameters)

    def test_returned_dataset_contains_result_metadata(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertTrue(fit.result.metadata.result.parameters)

    def test_returned_dataset_contains_data_metadata(self):
        self.create_dataset()
        self.dataset.id = 'foo'
        self.dataset.label = 'random spectral line'
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(self.dataset.id, fit.result.metadata.data.id)
        self.assertEqual(self.dataset.label, fit.result.metadata.data.label)

    def test_analyse_with_least_squares_algorithm(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        self.fit.parameters['algorithm']['method'] = 'least_squares'
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_analyse_with_algorithm_parameters(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        self.fit.parameters['algorithm']['parameters'] = {'xtol': 1e-6}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_analyse_with_unknown_algorithm_raises(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'start': 0}}
        self.fit.parameters['algorithm']['method'] = 'foobar'
        with self.assertRaisesRegex(ValueError, 'Unknown method'):
            self.dataset.analyse(self.fit)


class TestLHSFit(unittest.TestCase):
    def setUp(self):
        self.fit = fitpy.analysis.LHSFit()
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
        self.assertIn('Fit model to data of dataset '
                      'with LHS of starting conditions',
                      self.fit.description)

    def test_has_fit_parameter(self):
        self.assertTrue('fit' in self.fit.parameters)

    def test_has_algorithm_parameter(self):
        self.assertTrue('algorithm' in self.fit.parameters)

    def test_has_lhs_parameter(self):
        self.assertTrue('lhs' in self.fit.parameters)

    def test_analysis_returns_calculated_dataset(self):
        self.create_dataset()
        self.fit.model = self.model
        fit = self.dataset.analyse(self.fit)
        self.assertIsInstance(fit.result,
                              fitpy.dataset.CalculatedDatasetLHS)

    def test_analyse_returns_fitted_data(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_analyse_sets_algorithm_description(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertTrue(fit.parameters['algorithm']['description'])

    def test_returned_dataset_has_same_x_axis_as_model(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertListEqual(list(self.dataset.data.axes[0].values),
                             list(fit.result.data.axes[0].values))

    def test_returned_dataset_contains_residual(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertListEqual(list(np.zeros(len(self.dataset.data.data))),
                             list(fit.result.data.residual))

    def test_returned_dataset_contains_model_metadata(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(aspecd.utils.full_class_name(self.model),
                         fit.result.metadata.model.type)
        self.assertDictEqual(fit.model.parameters,
                             fit.result.metadata.model.parameters)

    def test_returned_dataset_contains_result_metadata(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertTrue(fit.result.metadata.result.parameters)

    def test_returned_dataset_contains_data_metadata(self):
        self.create_dataset()
        self.dataset.id = 'foo'
        self.dataset.label = 'random spectral line'
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(self.dataset.id, fit.result.metadata.data.id)
        self.assertEqual(self.dataset.label, fit.result.metadata.data.label)

    def test_returned_dataset_contains_lhs_metadata(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        fit = self.dataset.analyse(self.fit)
        self.assertTrue(fit.result.metadata.lhs.results)
        self.assertTrue(len(fit.result.metadata.lhs.samples))
        self.assertTrue(fit.result.metadata.lhs.discrepancy)

    def test_analyse_with_least_squares_algorithm(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        self.fit.parameters['algorithm']['method'] = 'least_squares'
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_analyse_with_algorithm_parameters(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        self.fit.parameters['algorithm']['parameters'] = {'xtol': 1e-6}
        fit = self.dataset.analyse(self.fit)
        self.assertEqual(2., self.dataset.data.axes[0].values[
            np.argmax(fit.result.data.data)])

    def test_analyse_with_unknown_algorithm_raises(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5}
        self.fit.parameters['algorithm']['method'] = 'foobar'
        with self.assertRaisesRegex(ValueError, 'Unknown method'):
            self.dataset.analyse(self.fit)

    def test_analyse_with_lhs_seed_returns_identical_samples(self):
        self.create_dataset()
        self.fit.model = self.model
        self.fit.parameters['fit'] = {'position': {'lhs_range': [-8, 8]}}
        self.fit.parameters['lhs'] = {'points': 5, 'rng_seed': 42}
        fit1 = self.dataset.analyse(self.fit)
        fit2 = self.dataset.analyse(self.fit)
        self.assertListEqual(
            list(fit1.result.metadata.lhs.samples.flatten()),
            list(fit2.result.metadata.lhs.samples.flatten())
        )
