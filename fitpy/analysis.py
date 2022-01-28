"""
Actual fitting in form of analysis steps derived from the ASpecD framework.

Fitting of a model to (experimental) data can always be seen as an analysis
step in context of the ASpecD framework, resulting in a calculated dataset.

"""

import aspecd.analysis
import lmfit
import numpy as np
from scipy.stats import qmc


class SimpleFit(aspecd.analysis.SingleAnalysisStep):
    # noinspection PyUnresolvedReferences
    """
    Perform basic fit of a model to data of a dataset.

    Attributes
    ----------
    model : :class:`aspecd.model.Model`
        Model to fit to the data of a dataset

    parameters : :class:`dict`
        All parameters necessary to perform the fit.

        These parameters will be available from the calculation metadata of
        the resulting :class:`fitpy.dataset.CalculatedDataset`.

        fit : :class:`dict`
            All model parameters that should be fitted.

            The keys of the dictionary need to correspond to the parameter
            names of the model that should be fitted. The values are dicts
            themselves, at least with the key ``start`` for the initial
            parameter value. Additionally, you may supply a ``range`` with
            a list as value defining the interval within the the parameter
            is allowed to vary during fitting.

        algorithm : :class:`dict`
            Settings of the algorithm used to fit the model to the data.

            The key ``method`` needs to correspond to the methods supported
            by :class:`lmfit.minimizer.Minimizer`.

            To provide more information independent on the naming of
            the respective methods in :mod:`lmfit.minimizer` and the
            corresponding :mod:`scipy.optimize` module, the key
            ``description`` contains a short description of the respective
            method.

            To pass additional parameters to the solver, use the
            ``parameters`` dict. Which parameters can be set depends on the
            actual solver. For details, see the :mod:`scipy.optimize`
            documentation.


    Raises
    ------
    ValueError
        Raised if the method provided in ``parameters['algorithm'][ 'method']``
        is not supported or invalid.


    Examples
    --------
    For convenience, a series of examples in recipe style (for details of
    the recipe-driven data analysis, see :mod:`aspecd.tasks`) is given below
    for how to make use of this class. The examples focus each on a single
    aspect.

    Fitting is always a two-step process: (i) define the model, and (ii)
    define the fitting task. Here and in the following examples we assume
    a dataset to be imported as ``dataset``, and the model is
    initially evaluated for this dataset (to get the same data dimensions
    and alike, see :mod:`aspecd.model` for details).

    .. note::

        Usually, you will have set another ASpecD-derived package as
        default package in your recipe for processing and analysing your data.
        Hence, you need to provide the package name (fitpy) in the ``kind``
        property, as shown in the examples.


    Suppose you have a dataset and want to fit a Gaussian to its data,
    in this case only varying the amplitude, but keeping position and
    width fixed to the values specified in the model:

    .. code-block:: yaml

        - kind: model
          type: Gaussian
          properties:
            parameters:
              position: 1.5
              width: 0.5
          from_dataset: dataset
          output: model
          result: gaussian_model

        - kind: fitpy.singleanalysis
          type: SimpleFit
          properties:
            model: gaussian_model
            parameters:
              fit:
                amplitude:
                  start: 5
          result: fitted_gaussian

    In this particular case, you define your model specifying position and
    width, and fit this to the data allowing only the parameter amplitude
    to vary, keeping position and width fixed at the given values.
    Furthermore, no range is provided for the values the amplitude can be
    varied.

    To provide a range (boundaries, interval) for the allowed values of a
    fit parameter, simply add the key ``range``:

    .. code-block:: yaml

        - kind: model
          type: Gaussian
          properties:
            parameters:
              position: 1.5
              width: 0.5
          from_dataset: dataset
          output: model
          result: gaussian_model

        - kind: fitpy.singleanalysis
          type: SimpleFit
          properties:
            model: gaussian_model
            parameters:
              fit:
                amplitude:
                  start: 5
                  range: [3, 7]
          result: fitted_gaussian

    Note that models usually will have standard values for all parameters.
    Therefore, you only need to define those parameters in the model task
    that shall *not* change during fitting and should have values
    different from the standard.

    If you were to fit multiple parameters of a model (as is usually the
    case), provide all these parameters in the fit section of the
    parameters of the fitting task:

    .. code-block:: yaml

        - kind: model
          type: Gaussian
          properties:
            parameters:
              width: 0.5
          from_dataset: dataset
          output: model
          result: gaussian_model

        - kind: fitpy.singleanalysis
          type: SimpleFit
          properties:
            model: gaussian_model
            parameters:
              fit:
                amplitude:
                  start: 5
                  range: [3, 7]
                position:
                  start: 2
                  range: [0, 4]
          result: fitted_gaussian

    While the default algorithm settings are quite sensible as a starting
    point, you can explicitly set the method and its parameters. Which
    parameters can be set depends on the method chosen, for details refer to
    the documentation of the underlying :mod:`scipy.optimize` module. The
    following example shows how to change the algorithm to ``least_squares``
    (using a Trust Region Reflective method) and to set the tolerance for
    termination by the change of the independent variables (``xtol`` parameter):

    .. code-block:: yaml

        - kind: model
          type: Gaussian
          properties:
            parameters:
              position: 1.5
              width: 0.5
          from_dataset: dataset
          output: model
          result: gaussian_model

        - kind: fitpy.singleanalysis
          type: SimpleFit
          properties:
            model: gaussian_model
            parameters:
              fit:
                amplitude:
                  start: 5
              algorithm:
                method: least_squares
                parameters:
                  xtol: 1e-6
          result: fitted_gaussian


    """

    def __init__(self):
        super().__init__()
        self.description = 'Fit model to data of dataset'
        self.model = None
        self.parameters['fit'] = {}
        self.parameters['algorithm'] = {
            'method': 'leastsq',
            'description': '',
            'parameters': {},
        }
        self.dataset_type = 'fitpy.dataset.CalculatedDataset'

        self._fit_parameters = lmfit.Parameters()
        self._fit_result = None
        self._method_descriptions = {
            'leastsq': 'Least-Squares minimization, using '
                       'Levenberg-Marquardt method',
            'least_squares': 'Least-Squares minimization, using Trust Region '
                             'Reflective method',
        }

    def _sanitise_parameters(self):
        if self.parameters['algorithm']['method'] \
                not in self._method_descriptions:
            message = 'Unknown method "{}"'.format(  # noqa
                self.parameters['algorithm']['method'])
            raise ValueError(message)

    def _perform_task(self):
        self.result = self.create_dataset()
        self.model.from_dataset(self.dataset)

        self.parameters['algorithm']['description'] = \
            self._method_descriptions[self.parameters['algorithm']['method']]

        self._prepare_fit_parameters()
        minimiser = lmfit.minimizer.Minimizer(self._calculate_residual,
                                              self._fit_parameters)
        self._fit_result = \
            minimiser.minimize(method=self.parameters['algorithm']['method'],
                               params=self._fit_parameters,
                               **self.parameters['algorithm']['parameters'])
        self._assign_fitted_model_to_result()

    def _prepare_fit_parameters(self):
        for key, value in self.model.parameters.items():
            parameter = lmfit.Parameter(name=key)
            if key in self.parameters['fit']:
                parameter.set(value=self.parameters['fit'][key]['start'])
                parameter.set(vary=True)
                if 'range' in self.parameters['fit'][key]:
                    parameter.set(min=self.parameters['fit'][key]['range'][0])
                    parameter.set(max=self.parameters['fit'][key]['range'][1])
            else:
                parameter.set(value=value)
                parameter.set(vary=False)
            self._fit_parameters.add(parameter)

    def _calculate_residual(self, parameters):
        self.model.parameters = parameters.valuesdict()
        residuals = self.dataset.data.data - self.model.evaluate()
        return residuals

    def _assign_fitted_model_to_result(self):
        self.model.parameters = self._fit_result.params.valuesdict()  # noqa
        model_dataset = self.model.create()
        self.result.data = model_dataset.data
        self.result.data.residual = self._fit_result.residual
        self.result.metadata.model.from_model(self.model)
        self.result.metadata.result.from_lmfit_minimizer_result(
            self._fit_result)
        self.result.metadata.data.from_dataset(self.dataset)


class LHSFit(aspecd.analysis.SingleAnalysisStep):
    # noinspection PyUnresolvedReferences
    """
    Fit of a model to data of a dataset using LHS of starting conditions.

    In case of more complicated fits, *e.g.* many variable parameters or a
    rough fitness landscape of the optimisation including several local
    minima, obtaining a robust fit and finding the global minimum requires
    to sample initial conditions and to perform fits for all these conditions.

    Here, a Latin Hypercube gets used to sample the initial conditions. For
    each of these, a fit is performed in the same way as in
    :class:`SimpleFit`. The best fit is stored in the result as usual,
    and additionally, the sample grid, the discrepancy as measure for the
    quality of the grid, as well as all results from the individual fits are
    stored in the ``lhs`` property of the metadata of the resulting dataset.
    This allows for both, handling this resulting dataset as usual and
    evaluating the robustness of the fit.

    Attributes
    ----------
    model : :class:`aspecd.model.Model`
        Model to fit to the data of a dataset

    parameters : :class:`dict`
        All parameters necessary to perform the fit.

        These parameters will be available from the calculation metadata of
        the resulting :class:`fitpy.dataset.CalculatedDatasetLHS`.

        fit : :class:`dict`
            All model parameters that should be fitted.

            The keys of the dictionary need to correspond to the parameter
            names of the model that should be fitted. The values are dicts
            themselves, at least with the key ``start`` for the initial
            parameter value. Additionally, you may supply a ``range`` with
            a list as value defining the interval within the the parameter
            is allowed to vary during fitting.

        algorithm : :class:`dict`
            Settings of the algorithm used to fit the model to the data.

            The key ``method`` needs to correspond to the methods supported
            by :class:`lmfit.minimizer.Minimizer`.

            To provide more information independent on the naming of
            the respective methods in :mod:`lmfit.minimizer` and the
            corresponding :mod:`scipy.optimize` module, the key
            ``description`` contains a short description of the respective
            method.

            To pass additional parameters to the solver, use the
            ``parameters`` dict. Which parameters can be set depends on the
            actual solver. For details, see the :mod:`scipy.optimize`
            documentation.

        lhs : :class:`dict`
            Settings for the Latin Hypercube used to sample initial conditions.


    Raises
    ------
    ValueError
        Raised if the method provided in ``parameters['algorithm'][ 'method']``
        is not supported or invalid.


    Examples
    --------
    For convenience, a series of examples in recipe style (for details of
    the recipe-driven data analysis, see :mod:`aspecd.tasks`) is given below
    for how to make use of this class. The examples focus each on a single
    aspect.

    Fitting is always a two-step process: (i) define the model, and (ii)
    define the fitting task. Here and in the following examples we assume
    a dataset to be imported as ``dataset``, and the model is
    initially evaluated for this dataset (to get the same data dimensions
    and alike, see :mod:`aspecd.model` for details).

    .. note::

        Usually, you will have set another ASpecD-derived package as
        default package in your recipe for processing and analysing your data.
        Hence, you need to provide the package name (fitpy) in the ``kind``
        property, as shown in the examples.


    Suppose you have a dataset and want to fit a Gaussian to its data,
    in this case only varying the amplitude, but keeping position and
    width fixed to the values specified in the model:

    .. code-block:: yaml

        - kind: model
          type: Gaussian
          properties:
            parameters:
              position: 1.5
              width: 0.5
          from_dataset: dataset
          output: model
          result: gaussian_model

        - kind: fitpy.singleanalysis
          type: LHSFit
          properties:
            model: gaussian_model
            parameters:
              fit:
                amplitude:
                  lhs_range: [2, 8]
              lhs:
                points = 7
          result: fitted_gaussian

    In this particular case, you define your model specifying position and
    width, and fit this to the data allowing only the parameter amplitude
    to vary, keeping position and width fixed at the given values.
    Furthermore, a range for the LHS for this parameter is provided, as well
    as the number of points sampled per dimension of the Latin Hypercube.

    """

    def __init__(self):
        super().__init__()
        self.description = 'Fit model to data of dataset ' \
                           'with LHS of starting conditions'
        self.model = None
        self.parameters['fit'] = {}
        self.parameters['algorithm'] = {
            'method': 'leastsq',
            'description': '',
            'parameters': {},
        }
        self.parameters['lhs'] = {'points': 1}
        self.dataset_type = 'fitpy.dataset.CalculatedDatasetLHS'

        self._method_descriptions = {
            'leastsq': 'Least-Squares minimization, using '
                       'Levenberg-Marquardt method',
            'least_squares': 'Least-Squares minimization, using Trust Region '
                             'Reflective method',
        }
        self._lhs_parameters = None
        self._lhs_samples = None
        self._lhs_discrepancy = None
        self._fit_parameters = []
        self._fit_results = []

    def _sanitise_parameters(self):
        if self.parameters['algorithm']['method'] not in \
                self._method_descriptions:
            message = 'Unknown method "{}"'.format(  # noqa
                self.parameters['algorithm']['method'])
            raise ValueError(message)

    def _perform_task(self):
        self.result = self.create_dataset()
        self.model.from_dataset(self.dataset)

        self.parameters['algorithm']['description'] = \
            self._method_descriptions[self.parameters['algorithm']['method']]

        self._create_lhs_parameters()
        self._create_lhs_samples()
        self._create_fit_parameters()

        self._perform_lhs_fit()

        self._assign_results_to_result()

    def _create_lhs_parameters(self):
        self._lhs_parameters = {
            key: value for key, value in self.parameters['fit'].items()
            if 'lhs_range' in value
        }

    def _create_lhs_samples(self):
        lhs_dimensions = len(self._lhs_parameters)
        sampler = qmc.LatinHypercube(d=lhs_dimensions)
        for key, value in self.parameters['lhs'].items():
            if hasattr(sampler, key):
                setattr(sampler, key, value)
        if sampler.rng_seed:
            sampler.reset()
        self._lhs_samples = sampler.random(self.parameters['lhs']['points'])
        self._lhs_discrepancy = qmc.discrepancy(self._lhs_samples)

        lower_bounds = []
        upper_bounds = []
        for value in self._lhs_parameters.values():
            lower_bounds.append(value['lhs_range'][0])
            upper_bounds.append(value['lhs_range'][1])
        self._lhs_samples = \
            qmc.scale(self._lhs_samples, lower_bounds, upper_bounds)

    def _create_fit_parameters(self):
        param_samples = {}
        for idx, key in enumerate(self._lhs_parameters.keys()):
            param_samples[key] = self._lhs_samples[:, idx]
        for point in range(self.parameters['lhs']['points']):
            self._fit_parameters.append(lmfit.Parameters())
            for key, value in self.model.parameters.items():
                parameter = lmfit.Parameter(name=key)
                if key in self.parameters['fit']:
                    if 'lhs_range' in self.parameters['fit'][key]:
                        parameter.set(value=param_samples[key][point])
                    else:
                        parameter.set(
                            value=self.parameters['fit'][key]['start'])
                    parameter.set(vary=True)
                    if 'range' in self.parameters['fit'][key]:
                        parameter.set(
                            min=self.parameters['fit'][key]['range'][0])
                        parameter.set(
                            max=self.parameters['fit'][key]['range'][1])
                else:
                    parameter.set(value=value)
                    parameter.set(vary=False)
                self._fit_parameters[point].add(parameter)

    def _calculate_residual(self, parameters):
        self.model.parameters = parameters.valuesdict()
        residuals = self.dataset.data.data - self.model.evaluate()
        return residuals

    def _perform_lhs_fit(self):
        for point in range(self.parameters['lhs']['points']):
            minimiser = lmfit.minimizer.Minimizer(self._calculate_residual,
                                                  self._fit_parameters[point])
            fit_result = minimiser.minimize(
                method=self.parameters['algorithm']['method'],
                params=self._fit_parameters[point],
                **self.parameters['algorithm']['parameters']
            )
            self._fit_results.append(fit_result)

    def _assign_results_to_result(self):
        chi_squares = [result.chisqr for result in self._fit_results]
        best_fit_index = np.argmin(chi_squares)
        best_fit = self._fit_results[best_fit_index]
        self.model.parameters = best_fit.params.valuesdict()  # noqa
        model_dataset = self.model.create()
        self.result.data = model_dataset.data

        self.result.data.residual = best_fit.residual
        self.result.metadata.model.from_model(self.model)
        self.result.metadata.result.from_lmfit_minimizer_result(best_fit)
        self.result.metadata.data.from_dataset(self.dataset)
        self.result.metadata.lhs.from_lmfit_minimizer_results(
            self._fit_results)
        self.result.metadata.lhs.samples = self._lhs_samples
        self.result.metadata.lhs.discrepancy = self._lhs_discrepancy
