"""
Actual fitting in form of analysis steps derived from the ASpecD framework.

.. sidebar:: Contents

    .. contents::
        :local:
        :depth: 1

Fitting of a model to (experimental) data can always be seen as an analysis
step in context of the ASpecD framework, resulting in a calculated dataset.


Introduction
============

Fitting in context of the FitPy framework is always a two-step process:

#. define the model, and

#. define the fitting task.

The model is an instance of :class:`aspecd.model.Model`, and the fitting
task one of the analysis steps contained in this module. They are, in turn,
instances of :class:`aspecd.analysis.AnalysisStep`.

A first, simple but complete example of a recipe performing a fit on
experimental data, is given below.

.. code-block:: yaml
    :linenos:

    format:
      type: ASpecD recipe
      version: '0.2'

    datasets:
      - /path/to/dataset

    tasks:
      - kind: model
        type: Gaussian
        properties:
          parameters:
            position: 1.5
            width: 0.5
        from_dataset: /path/to/dataset
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


In this case, a Gaussian model is created, with values for two parameters
set explicitly and not varied during the fit. The third parameter is varied
during the fit, within a given range. Furthermore, using
:class:`SimpleFit` here without further parameters, a least-squares fit
using the Levenberg-Marquardt method is carried out.


.. note::

    Usually, you will have set another ASpecD-derived package as
    default package in your recipe for processing and analysing your data.
    Hence, you need to provide the package name (fitpy) in the ``kind``
    property, as shown in the examples.


This seamless integration of FitPy into all packages derived from the ASpecD
framework ensures full reproducibility and allows to easily pre- and
postprocess the data accordingly. Particularly for analysing the results of
fits, have a look at the dedicated plotters in the :mod:`fitpy.plotting`
module and the reporters in the :mod:`fitpy.report` module.


Fitting strategies
==================

Fitting models to data is generally a complex endeavour, and FitPy will
*not* take any decisions for you. However, it provides powerful abstractions
and a simple user interface, letting you automate as much as possible,
while retaining full reproducibility. Thus, it is possible to create entire
pipelines spanning a series of different fitting strategies, analyse the
results, and making an informed decision for each individual question.

The following list provides an overview of the different fitting strategies
supported by FitPy (currently, as of January 2022, only a subset of these
strategies is implemented).

* Simple fitting of single datasets

  Make informed guesses for the initial values of the variable parameters of
  a model and fit the model to the data. The most straight-forward strategy.
  Still, different optimisation algorithms can be chosen.

  If the fitness landscape is rough and contains local minima, the fit may
  not converge or get stuck in local minima.

* Robust fitting via sampling of initial conditions (LHS)

  Instead of informed guesses for the initial values of the variable
  parameters of a model, these initial values are randomly chosen using a
  Latin Hypercube. For each of the resulting grid points, an optimisation is
  performed, analogous to what has been described above.

  Generally, this approach will take much longer, with the computing time
  scaling with the number of grid points, but it is much more robust,
  particularly with complicated fitness landscapes containing many local
  minima.

* Fitting multiple species to one dataset

  Basically the same as fitting a simple model to the data of a dataset,
  but this time providing a :class:`aspecd.model.CompositeModel`.

  Given the usually larger number of variable parameters, robust
  fitting strategies (LHS) should be used.

* Global fitting of several datasets at once

  Fit models with a joint set of parameters to a series of independent
  datasets. Can become arbitrarily complex given that some parameters may be
  allowed to independently vary for each dataset, while others are
  constrained, while still others (typically the majority) will be identical
  for each dataset.


Common to all these different fitting strategies is the need to sometimes omit
parts of a dataset from fitting.


Concrete fitting tasks implemented
==================================

Currently (as of January 2022), only fitting tasks are implemented that
operate on single datasets.

* :class:`SimpleFit`

  Perform basic fit of a model to data of a dataset.

  The result is stored as calculated dataset and can be investigated
  graphically using dedicated plotters from the :mod:`fitpy.plotting`
  module as well as reporters from the :mod:`fitpy.report` module.

  With default settings, a least-Squares minimization using the
  Levenberg-Marquardt method is carried out. Initial values and ranges for
  each variable parameter of the model can be specified, as well as
  details for the algorithm.

* :class:`LHSFit`

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


Helper classes
==============

Additionally to the fitting tasks described above, helper classes exist for
specific tasks.

* :class:`ExtractLHSStatistics`

  Extract statistical criterion from LHS results for evaluating robustness.

  When performing a robust fitting, *e.g.* by employing :class:`LHSFit`,
  evaluating the robustness of the obtained results is a crucial step.
  Therefore, the results from each individual fit starting with a grid
  point of the Latin Hypercube are contained in the resulting dataset.
  This analysis step extracts the given criterion from the calculated
  dataset and returns itself a calculated dataset with the values of the
  criterion sorted in ascending order as its data. The result can be
  graphically represented using a :class:`aspecd.plotting.SinglePlotter1D`.



Module documentation
====================

"""

import aspecd.analysis
import lmfit
import numpy as np
from scipy.stats import qmc


class SimpleFit(aspecd.analysis.SingleAnalysisStep):
    # noinspection PyUnresolvedReferences
    """
    Perform basic fit of a model to data of a dataset.

    The result is stored as calculated dataset and can be investigated
    graphically using dedicated plotters from the :mod:`fitpy.plotting`
    module as well as reporters from the :mod:`fitpy.report` module.

    With default settings, a least-Squares minimization using the
    Levenberg-Marquardt method is carried out. Initial values and ranges for
    each variable parameter of the model can be specified, as well as
    details for the algorithm.

    Attributes
    ----------
    result : :class:`fitpy.dataset.CalculatedDataset`
        Calculated dataset containing the result of the fit.

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
    result : :class:`fitpy.dataset.CalculatedDatasetLHS`
        Calculated dataset containing the result of the fit.

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

            The most important parameter is ``points``, defining the
            points in each direction of the Latin Hypercube.

            Additionally, all attributes of
            :class:`scipy.stats.qmc.LatinHypercube` can be set. Currently,
            the relevant parameters are ``centered`` (to center the point
            within the multi-dimensional grid) and ``rng_seed`` to allow for
            reproducible results.

            In case ``rng_seed`` is provided, the random number generator is
            reset and seeded with this value, ensuring reproducible creation
            of the grid.


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
                points: 7
          result: fitted_gaussian

    In this particular case, you define your model specifying position and
    width, and fit this to the data allowing only the parameter amplitude
    to vary, keeping position and width fixed at the given values.
    Furthermore, a range for the LHS for this parameter is provided, as well
    as the number of points sampled per dimension of the Latin Hypercube.

    Only those fitting parameters having set the ``lhs_range`` parameter
    will be used for sampling. All other parameters will be used with their
    starting values as defined:

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
                position:
                  start: 2
                  range: [0, 4]
              lhs:
                points: 7
          result: fitted_gaussian

    Here, only the ``amplitude`` parameter will be sampled (in this
    particular case resulting in a 1D Latin Hypercube), while for each of
    the grid points, the ``position`` parameter is set as given.

    Sometimes the grid created by the LHS should be reproducible. In this
    case, provide a seed for the random number generator used internally:

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
                points: 7
                rng_seed: 42
          result: fitted_gaussian

    Similarly, if the points should be centred within the multi-dimensional
    grid, set the ``centered`` property accordingly:

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
                points: 7
                centered: true
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
          type: LHSFit
          properties:
            model: gaussian_model
            parameters:
              fit:
                amplitude:
                  lhs_range: [2, 8]
              lhs:
                points: 7
              algorithm:
                method: least_squares
                parameters:
                  xtol: 1e-6
          result: fitted_gaussian

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
        self.parameters['lhs'] = {
            'points': 1,
            'centered': False,
            'rng_seed': None,
        }
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


class ExtractLHSStatistics(aspecd.analysis.SingleAnalysisStep):
    # noinspection PyUnresolvedReferences
    """
    Extract statistical criterion from LHS results for evaluating robustness.

    When performing a robust fitting, *e.g.* by employing :class:`LHSFit`,
    evaluating the robustness of the obtained results is a crucial step.
    Therefore, the results from each individual fit starting with a grid
    point of the Latin Hypercube are contained in the resulting dataset.
    This analysis step extracts the given criterion from the calculated
    dataset and returns itself a calculated dataset with the values of the
    criterion sorted in ascending order as its data. The result can be
    graphically represented using a :class:`aspecd.plotting.SinglePlotter1D`.

    Attributes
    ----------
    result : :class:`aspecd.dataset.CalculatedDataset`
        Calculated dataset containing the extracted statistical criterion.

    parameters : :class:`dict`
        All parameters necessary to perform the fit.

        These parameters will be available from the calculation metadata of
        the resulting :class:`fitpy.dataset.CalculatedDatasetLHS`.

        criterion : :class:`str`
            Statistical criterion extracted from the LHS results


    Examples
    --------
    For convenience, a series of examples in recipe style (for details of
    the recipe-driven data analysis, see :mod:`aspecd.tasks`) is given below
    for how to make use of this class. The examples focus each on a single
    aspect.

    Suppose you have fitted a Gaussian to the data of a dataset, as shown in
    the example section of the :class:`LHSFit` class. If you now want to
    extract the reduced chi square value and plot it, the whole procedure
    could look like this:

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
                points: 7
          result: fitted_gaussian

        - kind: fitpy.singleanalysis
          type: ExtractLHSStatistics
          properties:
            parameters:
              criterion: reduced_chi_square
          result: reduced_chi_squares
          apply_to: fitted_gaussian

        - kind: singleplot
          type: SinglePlotter1D
          properties:
            properties:
              drawing:
                marker: 'o'
                linestyle: 'none'
            filename: 'reduced_chi_squares.pdf'
          apply_to: reduced_chi_squares

    This would plot the reduced chi square values in ascending order,
    showing the individual values as not connected dots.

    """

    def __init__(self):
        super().__init__()
        self.description = 'Extract LHS statistics from calculated dataset'
        self.parameters['criterion'] = 'chi_square'

        self._criterion_names = {
            'chi_square': 'chi square',
            'reduced_chi_square': 'reduced chi square',
            'akaike_information_criterion': 'Akaike information criterion',
            'bayesian_information_criterion': 'Bayesian information criterion',
        }

    @staticmethod
    def applicable(dataset):
        """Check whether analysis step is applicable to the given dataset.

        Returns
        -------
        applicable : :class:`bool`
            `True` if successful, `False` otherwise.

        """
        return hasattr(dataset.metadata, 'lhs')

    def _perform_task(self):
        self.result = self.create_dataset()
        criterion = [getattr(result, self.parameters['criterion'])
                     for result in self.dataset.metadata.lhs.results]
        criterion.sort()
        self.result.data.data = np.asarray(criterion)
        self.result.data.axes[0].quantity = 'index of samples'
        self.result.data.axes[1].quantity = \
            self._criterion_names[self.parameters['criterion']]
