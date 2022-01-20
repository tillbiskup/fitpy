import unittest

import numpy as np

import fitpy.dataset


class TestCalculatedDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = fitpy.dataset.CalculatedDataset()

    def test_instantiate_class(self):
        pass

    def test_data_has_residual_property(self):
        self.assertTrue(hasattr(self.dataset.data, 'residual'))


class TestData(unittest.TestCase):
    def setUp(self):
        self.data = fitpy.dataset.Data()

    def test_instantiate_class(self):
        pass

    def test_has_residual_property(self):
        self.assertTrue(hasattr(self.data, 'residual'))

    def test_set_residual_with_shape_unequal_data_shape_raises(self):
        message = 'Shapes of data and residual need to match'
        with self.assertRaisesRegex(ValueError, message):
            self.data.residual = np.zeros(5)

    def test_set_residual(self):
        self.data.data = np.zeros(5)
        self.data.residual = np.zeros(5)
        self.assertListEqual(list(np.zeros(5)), list(self.data.residual))

    def test_residual_in_dict(self):
        dict_ = self.data.to_dict()
        self.assertIn('residual', dict_)
