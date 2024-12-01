import unittest

import aspecd.model
import lmfit
import numpy as np

import fitpy.dataset


class TestCalculatedDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = fitpy.dataset.CalculatedDataset()

    def test_instantiate_class(self):
        pass

    def test_data_has_residual_property(self):
        self.assertTrue(hasattr(self.dataset.data, "residual"))

    def test_data_calculated_is_true(self):
        self.assertTrue(self.dataset.data.calculated)

    def test_origdata_has_residual_property(self):
        self.assertTrue(hasattr(self.dataset._origdata, "residual"))

    def test_origdata_calculated_is_true(self):
        self.assertTrue(self.dataset._origdata.calculated)

    def test_metadata_has_result_property(self):
        self.assertTrue(hasattr(self.dataset.metadata, "result"))


class TestCalculatedDatasetLHS(unittest.TestCase):
    def setUp(self):
        self.dataset = fitpy.dataset.CalculatedDatasetLHS()

    def test_instantiate_class(self):
        pass

    def test_data_has_residual_property(self):
        self.assertTrue(hasattr(self.dataset.data, "residual"))

    def test_data_calculated_is_true(self):
        self.assertTrue(self.dataset.data.calculated)

    def test_origdata_has_residual_property(self):
        self.assertTrue(hasattr(self.dataset._origdata, "residual"))

    def test_origdata_calculated_is_true(self):
        self.assertTrue(self.dataset._origdata.calculated)

    def test_metadata_has_result_property(self):
        self.assertTrue(hasattr(self.dataset.metadata, "lhs"))


class TestData(unittest.TestCase):
    def setUp(self):
        self.data = fitpy.dataset.Data()

    def test_instantiate_class(self):
        pass

    def test_has_residual_property(self):
        self.assertTrue(hasattr(self.data, "residual"))

    def test_set_residual_with_shape_unequal_data_shape_raises(self):
        message = "Shapes of data and residual need to match"
        with self.assertRaisesRegex(ValueError, message):
            self.data.residual = np.zeros(5)

    def test_set_residual(self):
        self.data.data = np.zeros(5)
        self.data.residual = np.zeros(5)
        self.assertListEqual(list(np.zeros(5)), list(self.data.residual))

    def test_residual_in_dict(self):
        dict_ = self.data.to_dict()
        self.assertIn("residual", dict_)


class TestCalculatedDatasetMetadata(unittest.TestCase):
    def setUp(self):
        self.metadata = fitpy.dataset.CalculatedDatasetMetadata()

    def test_instantiate_class(self):
        pass

    def test_has_model_property(self):
        self.assertTrue(hasattr(self.metadata, "model"))

    def test_has_data_property(self):
        self.assertTrue(hasattr(self.metadata, "data"))

    def test_has_result_property(self):
        self.assertTrue(hasattr(self.metadata, "result"))


class TestModel(unittest.TestCase):
    def setUp(self):
        self.metadata = fitpy.dataset.Model()

    def test_instantiate_class(self):
        pass

    def test_has_type_property(self):
        self.assertTrue(hasattr(self.metadata, "type"))

    def test_has_parameters_property(self):
        self.assertTrue(hasattr(self.metadata, "parameters"))

    def test_from_model_sets_type(self):
        model = aspecd.model.Gaussian()
        self.metadata.from_model(model)
        self.assertEqual("aspecd.model.Gaussian", self.metadata.type)

    def test_from_model_sets_parameters(self):
        model = aspecd.model.Gaussian()
        self.metadata.from_model(model)
        self.assertDictEqual(model.parameters, self.metadata.parameters)


class TestDataMetadata(unittest.TestCase):
    def setUp(self):
        self.metadata = fitpy.dataset.DataMetadata()

    def test_instantiate_class(self):
        pass

    def test_has_id_property(self):
        self.assertTrue(hasattr(self.metadata, "id"))

    def test_has_label_property(self):
        self.assertTrue(hasattr(self.metadata, "label"))

    def test_from_dataset_sets_id(self):
        dataset = fitpy.dataset.CalculatedDataset()
        dataset.id = "foo"
        self.metadata.from_dataset(dataset)
        self.assertEqual(dataset.id, self.metadata.id)

    def test_from_dataset_sets_label(self):
        dataset = fitpy.dataset.CalculatedDataset()
        dataset.label = "bar"
        self.metadata.from_dataset(dataset)
        self.assertEqual(dataset.label, self.metadata.label)


class TestResult(unittest.TestCase):
    def setUp(self):
        self.metadata = fitpy.dataset.Result()
        self.result = lmfit.minimizer.MinimizerResult()

    def perform_fit(self):
        p_true = lmfit.Parameters()
        p_true.add("amp", value=14.0)
        p_true.add("period", value=5.46)
        p_true.add("shift", value=0.123)
        p_true.add("decay", value=0.032)

        def residual(pars, x, data=None):
            """Model a decaying sine wave and subtract data."""
            vals = pars.valuesdict()

            if abs(vals["shift"]) > np.pi / 2:
                vals["shift"] = vals["shift"] - np.sign(vals["shift"]) * np.pi
            model = (
                vals["amp"]
                * np.sin(vals["shift"] + x / vals["period"])
                * np.exp(-x * x * vals["decay"] * vals["decay"])
            )
            if data is None:
                return model
            return model - data

        np.random.seed(0)
        x = np.linspace(0.0, 250.0, 1001)
        noise = np.random.normal(scale=0.7215, size=x.size)
        data_ = residual(p_true, x) + noise

        fit_params = lmfit.Parameters()
        fit_params.add("amp", value=13.0)
        fit_params.add("period", value=2)
        fit_params.add("shift", value=0.0)
        fit_params.add("decay", value=0.02)

        self.result = lmfit.minimize(
            residual, fit_params, args=(x,), kws={"data": data_}
        )

    def test_instantiate_class(self):
        pass

    def test_from_lmfit_minimizer_result_sets_attributes(self):
        self.perform_fit()
        self.metadata.from_lmfit_minimizer_result(self.result)
        mappings = {
            "params": "parameters",
            "success": "success",
            "errorbars": "error_bars",
            "nfev": "n_function_evaluations",
            "nvarys": "n_variables",
            "nfree": "degrees_of_freedom",
            "chisqr": "chi_square",
            "redchi": "reduced_chi_square",
            "aic": "akaike_information_criterion",
            "bic": "bayesian_information_criterion",
            "var_names": "variable_names",
            "covar": "covariance_matrix",
            "init_vals": "initial_values",
            "message": "message",
        }
        for key, value in mappings.items():
            if isinstance(getattr(self.result, key), list):
                self.assertListEqual(
                    list(getattr(self.result, key)),
                    list(getattr(self.metadata, value)),
                )
            elif isinstance(getattr(self.result, key), np.ndarray):
                self.assertListEqual(
                    list(getattr(self.result, key).flatten()),
                    list(getattr(self.metadata, value).flatten()),
                )
            else:
                self.assertEqual(
                    getattr(self.result, key), getattr(self.metadata, value)
                )

    def test_to_dict_adds_value_to_parameters(self):
        self.perform_fit()
        self.metadata.from_lmfit_minimizer_result(self.result)
        dict_ = self.metadata.to_dict()
        for key in dict_["parameters"].keys():
            self.assertIn("value", dict_["parameters"][key])


class TestLHS(unittest.TestCase):
    def setUp(self):
        self.metadata = fitpy.dataset.LHS()
        self.result = lmfit.minimizer.MinimizerResult()

    def perform_fit(self):
        p_true = lmfit.Parameters()
        p_true.add("amp", value=14.0)
        p_true.add("period", value=5.46)
        p_true.add("shift", value=0.123)
        p_true.add("decay", value=0.032)

        def residual(pars, x, data=None):
            """Model a decaying sine wave and subtract data."""
            vals = pars.valuesdict()

            if abs(vals["shift"]) > np.pi / 2:
                vals["shift"] = vals["shift"] - np.sign(vals["shift"]) * np.pi
            model = (
                vals["amp"]
                * np.sin(vals["shift"] + x / vals["period"])
                * np.exp(-x * x * vals["decay"] * vals["decay"])
            )
            if data is None:
                return model
            return model - data

        np.random.seed(0)
        x = np.linspace(0.0, 250.0, 1001)
        noise = np.random.normal(scale=0.7215, size=x.size)
        data_ = residual(p_true, x) + noise

        fit_params = lmfit.Parameters()
        fit_params.add("amp", value=13.0)
        fit_params.add("period", value=2)
        fit_params.add("shift", value=0.0)
        fit_params.add("decay", value=0.02)

        self.result = lmfit.minimize(
            residual, fit_params, args=(x,), kws={"data": data_}
        )

    def test_instantiate_class(self):
        pass

    def test_has_samples_property(self):
        self.assertTrue(hasattr(self.metadata, "samples"))

    def test_has_discrepancy_property(self):
        self.assertTrue(hasattr(self.metadata, "discrepancy"))

    def test_has_results_property(self):
        self.assertTrue(hasattr(self.metadata, "results"))

    def test_from_lmfit_minimizer_results_sets_results(self):
        self.perform_fit()
        self.metadata.from_lmfit_minimizer_results([self.result])

        mappings = {
            "params": "parameters",
            "success": "success",
            "errorbars": "error_bars",
            "nfev": "n_function_evaluations",
            "nvarys": "n_variables",
            "nfree": "degrees_of_freedom",
            "chisqr": "chi_square",
            "redchi": "reduced_chi_square",
            "aic": "akaike_information_criterion",
            "bic": "bayesian_information_criterion",
            "var_names": "variable_names",
            "covar": "covariance_matrix",
            "init_vals": "initial_values",
            "message": "message",
        }
        metadata = self.metadata.results[0]
        for key, value in mappings.items():
            if isinstance(getattr(self.result, key), list):
                self.assertListEqual(
                    list(getattr(self.result, key)),
                    list(getattr(metadata, value)),
                )
            elif isinstance(getattr(self.result, key), np.ndarray):
                self.assertListEqual(
                    list(getattr(self.result, key).flatten()),
                    list(getattr(metadata, value).flatten()),
                )
            else:
                self.assertEqual(
                    getattr(self.result, key), getattr(metadata, value)
                )
