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
import copy

import aspecd.dataset
import aspecd.metadata
import aspecd.utils
import numpy as np


class CalculatedDataset(aspecd.dataset.CalculatedDataset):
    """
    Dataset containing results of fitting a model to data.

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


class CalculatedDatasetLHS(CalculatedDataset):
    # noinspection PyUnresolvedReferences
    """
    Dataset containing results of fitting a model to data.

    Attributes
    ----------
    data : :class:`Data`
        numeric data, residual, and axes

        In contrast to other datasets, it contains the residual
        (difference between fitted model and original data) as well.

    metadata : :obj:`CalculatedDatasetLHSMetadata`
        hierarchical key-value store of metadata

    """

    def __init__(self):
        super().__init__()
        self.metadata = CalculatedDatasetLHSMetadata()


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
    # noinspection PyUnresolvedReferences
    """
    Metadata for a dataset with calculated data.

    This class contains the metadata for a dataset consisting of
    calculated data, i.e., :class:`CalculatedDataset`.

    Metadata can be converted to dict via
    :meth:`aspecd.utils.ToDictMixin.to_dict()`, e.g., for generating
    reports using templates and template engines.

    Attributes
    ----------
    calculation : :class:`aspecd.metadata.Calculation`
        Information on the calculation.

        Contain, *inter alia*, the parameters of the calculation.

    model : :class:`Model`
        Details of the model fitted to the data

    data : class:`DataMetadata`
        Details of the data the model has been fitted to

    result : :class:`Result`
        Summary of results of fit

    """

    def __init__(self):
        super().__init__()
        self.model = Model()
        self.data = DataMetadata()
        self.result = Result()


class CalculatedDatasetLHSMetadata(CalculatedDatasetMetadata):
    # noinspection PyUnresolvedReferences
    """
    Metadata for a dataset with calculated data.

    This class contains the metadata for a dataset consisting of
    calculated data, i.e., :class:`CalculatedDataset`.

    Metadata can be converted to dict via
    :meth:`aspecd.utils.ToDictMixin.to_dict()`, e.g., for generating
    reports using templates and template engines.

    Attributes
    ----------
    calculation : :class:`aspecd.metadata.Calculation`
        Information on the calculation.

        Contain, *inter alia*, the parameters of the calculation.

    model : :class:`Model`
        Details of the model fitted to the data

    data : class:`DataMetadata`
        Details of the data the model has been fitted to

    result : :class:`Result`
        Summary of results of fit

    lhs : :class:`LHS`
        Details of the LHS and its full results for each sampling

    """

    def __init__(self):
        super().__init__()
        self.lhs = LHS()


class Model(aspecd.metadata.Metadata):
    """
    Metadata of a model fitted to data of a dataset.

    Part of the metadata of a :class:`CalculatedDataset` containing the
    data of the model fitted to the data of another (experimental) dataset.

    Attributes
    ----------
    type : :class:`str`
        Full class name (including package) of the respective model class

    parameters : :class:`dict`
        All parameters of the model

    """

    def __init__(self):
        super().__init__()
        self.type = ''
        self.parameters = {}

    def from_model(self, model):
        """
        Set attributes from :class:`aspecd.model.Model`.

        Parameters
        ----------
        model : :class:`aspecd.model.Model`
            Model the attributes should be obtained from

        """
        self.type = aspecd.utils.full_class_name(model)
        self.parameters = copy.deepcopy(model.parameters)


class DataMetadata(aspecd.metadata.Metadata):
    """
    Metadata of the data(set) a model has been fitted to.

    Part of the metadata of a :class:`CalculatedDataset` containing information
    of the (experimental) dataset the model has been fitted to.

    Attributes
    ----------
    id : :class:`str`
        (unique) identifier of the dataset (i.e., path, LOI, or else)

    label : :class:`str`
        Short description of the dataset

        Can be set by the user, defaults to the value set as
        :attr:`aspecd.dataset.Dataset.id` by the importer.

    """

    def __init__(self):
        super().__init__()
        self.id = ''  # noqa
        self.label = ''

    def from_dataset(self, dataset):
        """
        Set attributes from :class:`aspecd.dataset.Dataset`.

        Parameters
        ----------
        dataset : :class:`aspecd.dataset.Dataset`
            Dataset the attributes should be obtained from

        """
        self.id = dataset.id
        self.label = dataset.label


class Result(aspecd.metadata.Metadata):
    """
    Metadata of results of fitting a model to data of a dataset.

    Part of the metadata of a :class:`CalculatedDataset` containing the
    data of the model fitted to the data of another (experimental) dataset.

    While resembling the structure of the
    :class:`lmfit.minimizer.MinimizerResult` class, this class tries to
    abstract away from the attributes in terms of their names and
    introduces more readable (and more lengthily) attribute names.

    Attributes
    ----------
    parameters : :class:`lmfit.parameter.Parameters`
        The best-fit parameters resulting from the fit.

    success : :class:`bool`
        True if the fit succeeded, otherwise False.

    error_bars : :class:`bool`
        True if uncertainties were estimated, otherwise False.

    n_function_evaluations : :class:`int`
        Number of function evaluations

    n_variables : :class:`int`
        Number of variables of the model

    degrees_of_freedom : :class:`int`
        Degrees of freedom

    chi_square : :class:`float`
        Chi-square value

        For this value to be meaningful, the residual function needs to be
        scaled properly to the uncertainties in the data.

    reduced_chi_square : :class:`float`
        Reduced chi-square value

        For this value to be meaningful, the residual function needs to be
        scaled properly to the uncertainties in the data.

    akaike_information_criterion : :class:`float`
        Akaike Information Criterion statistic

    bayesian_information_criterion : :class:`float`
        Bayesian Information Criterion statistic

    variable_names : :class:`list`
        Ordered list of variable parameter names used in the optimisation.

    covariance_matrix : :class:`numpy.ndarray`
        Covariance matrix from minimisation.

        Rows and columns correspond to :attr:`variable_names`

    initial_values : :class:`list`
        List of initial values for variable parameters.

        For the corresponding parameter names see :attr:`variable_names`.

    message : :class:`str`
        Message regarding the fit success.

    """

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

    def to_dict(self, remove_empty=False):
        """
        Create dictionary containing public attributes of an object.

        Parameters
        ----------
        remove_empty : :class:`bool`
            Whether to remove keys with empty values

            Default: False

        Returns
        -------
        public_attributes : :class:`collections.OrderedDict`
            Ordered dictionary containing the public attributes of the object

            The order of attribute definition is preserved

        """
        dict_ = super().to_dict(remove_empty=remove_empty)
        if self.parameters:
            parameter_values = self.parameters.valuesdict()
            for key in dict_['parameters'].keys():
                dict_['parameters'][key]['value'] = parameter_values[key]
        return dict_

    def from_lmfit_minimizer_result(self, result):
        """
        Set attributes from :class:`lmfit.minimizer.MinimizerResult`.

        Parameters
        ----------
        result : :class:`lmfit.minimizer.MinimizerResult`
            Result of a minimisation using lmfit

        """
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


class LHS(aspecd.metadata.Metadata):
    """
    Metadata of the LHS and its full results for each sampling.

    Part of the metadata of a :class:`CalculatedDatasetLHS` containing the
    data of the model fitted to the data of another (experimental) dataset.

    Attributes
    ----------
    samples : :class:`numpy.array`
        Grid of the Latin Hypercube

    discrepancy : :class:`float`
        Discrepancy of the sample.

        The discrepancy is a uniformity criterion used to assess the space
        filling of a number of samples in a hypercube. A discrepancy
        quantifies the distance between the continuous uniform distribution
        on a hypercube and the discrete uniform distribution on distinct
        sample points. (from :func:`scipy.stats.qmc.discrepancy`)

    results : :class:`list`
        Results for each sample of the Latin Hypercube.

        Each result is an instance of :class:`Result`.

    """

    def __init__(self):
        super().__init__()
        self.samples = None
        self.discrepancy = None
        self.results = []

    def from_lmfit_minimizer_results(self, results):
        """
        Set attributes from :class:`lmfit.minimizer.MinimizerResult`.

        Parameters
        ----------
        results : :class:`list`
            List of results of a minimisation using lmfit

            Each result is an instance of
            :class:`lmfit.minimizer.MinimizerResult` and gets transferred to
            an instance of :class:`Result`.

        """
        for result in results:
            metadata = Result()
            metadata.from_lmfit_minimizer_result(result)
            self.results.append(metadata)
