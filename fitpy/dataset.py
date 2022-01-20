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


class CalculatedDataset(aspecd.dataset.CalculatedDataset):
    """
    Datasets containing results of fitting a model to data.

    Attributes
    ----------
    data : :class:`Data`
        numeric data, residual, and axes

        In contrast to other datasets, it contains the residual
        (difference between fitted model and original data) as well.

    """
    def __init__(self):
        super().__init__()
        self.data = Data()


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
