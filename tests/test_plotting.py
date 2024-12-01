import unittest

import matplotlib.pyplot as plt
import numpy as np

import aspecd.dataset
from aspecd.exceptions import NotApplicableToDatasetError
import aspecd.plotting

import fitpy.dataset
import fitpy.plotting


class TestSinglePlotter1D(unittest.TestCase):
    def setUp(self):
        self.plotter = fitpy.plotting.SinglePlotter1D()
        self.dataset = fitpy.dataset.CalculatedDataset()

    def tearDown(self):
        plt.close()

    def create_dataset(self):
        npoints = 100
        self.dataset.data.data = np.linspace(0, 1, npoints)
        self.dataset.data.residual = np.random.random(npoints) - 0.5

    def test_instantiate_class(self):
        pass

    def test_apply_to_dataset_without_residual_raises(self):
        with self.assertRaises(NotApplicableToDatasetError):
            dataset = aspecd.dataset.CalculatedDataset()
            dataset.plot(self.plotter)

    def test_apply_to_dataset_with_residual(self):
        self.create_dataset()
        self.dataset.plot(self.plotter)

    def test_apply_to_2D_dataset_with_residual_raises(self):
        with self.assertRaises(NotApplicableToDatasetError):
            dataset = fitpy.dataset.CalculatedDataset()
            dataset.data.data = np.zeros([5, 5])
            dataset.plot(self.plotter)

    def test_plot_sets_data_property(self):
        self.create_dataset()
        plot = self.dataset.plot(self.plotter)
        self.assertTrue(plot.data)

    def test_plot_shows_model_and_original_data(self):
        self.create_dataset()
        plot = self.dataset.plot(self.plotter)
        self.assertListEqual(
            list(plot.data_drawing.get_data()[1]),
            list(self.dataset.data.data + self.dataset.data.residual),
        )

    def test_plot_sets_label_for_fitted_model(self):
        self.create_dataset()
        plot = self.dataset.plot(self.plotter)
        self.assertEqual("fit", plot.properties.drawing.label)

    def test_plot_with_label_set_via_properties(self):
        self.create_dataset()
        self.plotter.properties.drawing.label = "foo"
        plot = self.dataset.plot(self.plotter)
        self.assertEqual(
            self.plotter.properties.drawing.label,
            plot.properties.drawing.label,
        )

    def test_plot_sets_label_of_data(self):
        self.create_dataset()
        plot = self.dataset.plot(self.plotter)
        properties = fitpy.plotting.SinglePlot1DProperties()
        self.assertEqual(properties.data.label, plot.data_drawing.get_label())

    def test_plot_sets_color_of_data(self):
        self.create_dataset()
        plot = self.dataset.plot(self.plotter)
        self.assertEqual(
            plot.properties.drawing.color, plot.drawing.get_color()
        )

        # plt.show()


class TestSinglePlot1DProperties(unittest.TestCase):
    def setUp(self):
        self.plot_properties = fitpy.plotting.SinglePlot1DProperties()

    def test_instantiate_class(self):
        pass

    def test_has_data_properties(self):
        self.assertTrue(hasattr(self.plot_properties, "data"))
        self.assertIsInstance(
            self.plot_properties.data, aspecd.plotting.LineProperties
        )

    def test_data_property_has_sensible_default_label(self):
        self.assertEqual("data", self.plot_properties.data.label)

    def test_data_property_has_sensible_default_color(self):
        self.assertEqual("#999", self.plot_properties.data.color)
