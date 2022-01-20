"""
Actual fitting in form of analysis steps derived from the ASpecD framework.

Fitting of a model to (experimental) data can always be seen as an analysis
step in context of the ASpecD framework, resulting in a calculated dataset.

"""

import lmfit

import aspecd.analysis


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

        fit : :class:`dict`
            All model parameters that should be fitted.

            The keys of the dictionary need to correspond to the parameter
            names of the model that should be fitted. The values are dicts
            themselves, at least with the key ``start`` for the initial
            parameter value. Additionally, you may supply a ``range`` with
            a list as value defining the interval within the the parameter
            is allowed to vary during fitting.


    Examples
    --------
    For convenience, a series of examples in recipe style (for details of
    the recipe-driven data analysis, see :mod:`aspecd.tasks`) is given below
    for how to make use of this class. The examples focus each on a single
    aspect.

    Suppose you have a dataset and want to fit a Gaussian to its data.
    Fitting is always a two-step process: (i) define the model, and (ii)
    define the fitting task. Here and in the following examples we assume
    a dataset to be imported as ``dataset``, and the model is
    initially evaluated for this dataset (to get the same data dimensions
    and alike, see :mod:`aspecd.model` for details).

    .. code-block:: yaml

        - kind: model
          type: Gaussian
          properties:
            parameters:
              position: 1.5
              width: 0.5
          from_dataset: dataset
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


    """

    def __init__(self):
        super().__init__()
        self.description = 'Fit model to data of dataset'
        self.model = None
        self.parameters['fit'] = dict()

        self._fit_parameters = lmfit.Parameters()
        self._fit_result = None

    def _perform_task(self):
        self.result = self.create_dataset()
        self.model.from_dataset(self.dataset)

        self._prepare_fit_parameters()
        self._fit_result = lmfit.minimize(self._calculate_residual,
                                          self._fit_parameters)
        self._assign_fitted_model_to_result()

    def _prepare_fit_parameters(self):
        for key, value in self.model.parameters.items():
            parameter = lmfit.Parameter(name=key)
            if key in self.parameters['fit']:
                parameter.set(value=self.parameters['fit'][key]['start'])
                parameter.set(vary=True)
            else:
                parameter.set(value=value)
                parameter.set(vary=False)
            self._fit_parameters.add(parameter)

    def _calculate_residual(self, parameters):
        self.model.parameters = parameters.valuesdict()
        tmp_dataset = self.model.create()
        residuals = self.dataset.data.data - tmp_dataset.data.data
        return residuals

    def _assign_fitted_model_to_result(self):
        self.model.parameters = self._fit_result.params.valuesdict()  # noqa
        model_dataset = self.model.create()
        self.result.data = model_dataset.data
