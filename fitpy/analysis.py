"""
Actual fitting in form of analysis steps derived from the ASpecD framework.

Fitting of a model to (experimental) data can always be seen as an analysis
step in context of the ASpecD framework, resulting in a calculated dataset.

"""

import lmfit

import aspecd.analysis


class SimpleFit(aspecd.analysis.SingleAnalysisStep):

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
