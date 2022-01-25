"""Datasets: units containing data and metadata.

The dataset is one key concept of the ASpecD framework and hence the FitPy
package derived from it, consisting of the data as well as the corresponding
metadata. Storing metadata in a structured way is a prerequisite for a
semantic understanding within the routines. Furthermore, a history of every
processing, analysis and annotation step is recorded as well, aiming at a
maximum of reproducibility. This is part of how the ASpecD framework and
therefore the FitPy package tries to support good scientific practice.

Therefore, each processing and analysis step of data should always be
performed using the respective methods of a dataset, at least as long as it
can be performed on a single dataset.

The reason for providing an own class for calculated datasets is to ensure
a consistent handling of fit results and provide means for specialised
plotters available in the :mod:`fitpy.plotting` module to check for their
applicability.

"""

import aspecd.dataset
import aspecd.metadata
import numpy as np


class CalculatedDataset(aspecd.dataset.CalculatedDataset):
    """
    Datasets containing results of fitting a model to data.

    Attributes
    ----------
    data : :class:`Data`
        numeric data, residual, and axes

        In contrast to other datasets, it contains the residual
        (difference between fitted model and original data) as well.

    metadata : :obj:`CalculatedDatasetMetadata`
        hierarchical key-value store of metadata

    """
    def __init__(self):
        super().__init__()
        self.data = Data()
        self.data.calculated = True
        self._origdata = Data()
        self._origdata.calculated = True
        self.metadata = CalculatedDatasetMetadata()


class Data(aspecd.dataset.Data):
    """
    Unit containing numeric data, residual, and corresponding axes.

    In contrast to the base class of the ASpecD framework, it contains the
    residual (difference between fitted model and original data) as well.

    """

    def __init__(self):
        super().__init__()
        self._residual = self._data
        self._include_in_to_dict.append('residual')

    @property
    def residual(self):
        """
        Residual (difference between fitted model and original data).

        A residual need always to have the same shape as the corresponding
        data. If you try to set a residual not conforming to this
        condition, a :class:`ValueError` will be raised.
        """
        return self._residual

    @residual.setter
    def residual(self, residual):
        if residual.shape != self.data.shape:
            raise ValueError('Shapes of data and residual need to match.')
        self._residual = residual


class CalculatedDatasetMetadata(aspecd.metadata.CalculatedDatasetMetadata):
    """
    Metadata for a dataset with calculated data.

    This class contains the metadata for a dataset consisting of
    calculated data, i.e., :class:`CalculatedDataset`.

    Metadata can be converted to dict via
    :meth:`aspecd.utils.ToDictMixin.to_dict()`, e.g., for generating
    reports using templates and template engines.

    Attributes
    ----------
    result : :class:`Result`
        Summary of results of fit

    """

    def __init__(self):
        super().__init__()
        self.result = Result()


class Result(aspecd.metadata.Metadata):

    def __init__(self):
        super().__init__()
        self.parameters = None
        self.success = False
        self.error_bars = False
        self.n_function_evaluations = 0
        self.n_variables = 0
        self.degrees_of_freedom = 0
        self.chi_square = 0.
        self.reduced_chi_square = 0.
        self.akaike_information_criterion = 0.
        self.bayesian_information_criterion = 0.
        self.variable_names = []
        self.covariance_matrix = np.ndarray([0])
        self.initial_values = []
        self.message = ''

    def from_lmfit_minimizer_result(self, result):
        mappings = {
            'params': 'parameters',
            'success': 'success',
            'errorbars': 'error_bars',
            'nfev': 'n_function_evaluations',
            'nvarys': 'n_variables',
            'nfree': 'degrees_of_freedom',
            'chisqr': 'chi_square',
            'redchi': 'reduced_chi_square',
            'aic': 'akaike_information_criterion',
            'bic': 'bayesian_information_criterion',
            'var_names': 'variable_names',
            'covar': 'covariance_matrix',
            'init_vals': 'initial_values',
            'message': 'message',
        }
        for key, value in mappings.items():
            if hasattr(result, key):
                setattr(self, value, getattr(result, key))
