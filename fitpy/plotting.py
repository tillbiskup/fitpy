"""
Graphical presentation of the results of fitting models to datasets.

Fitting itself is a quite complicated process, and it is crucial for routine
use to have graphical representations of the results, besides
automatically generated reports that usually contain such
graphical
representations.

"""

import aspecd.plotting


class SinglePlotter1D(aspecd.plotting.SinglePlotter1D):
    # noinspection PyUnresolvedReferences
    """1D plots of single datasets, including original data and fitted model.

    Convenience class taking care of 1D plots of single datasets. The type
    of plot can be set in its :attr:`aspecd.plotting.SinglePlotter1D.type`
    attribute. Allowed types are stored in the
    :attr:`aspecd.plotting.SinglePlotter1D.allowed_types` attribute.


    Attributes
    ----------
    properties : :class:`SinglePlot1DProperties`
        Properties of the plot, defining its appearance

        For the properties that can be set this way, see the documentation
        of the :class:`fitpy.plotting.SinglePlot1DProperties` and
        :class:`aspecd.plotting.SinglePlot1DProperties` classes.

    data : :class:`matplotlib.artist.Artist`
        Actual graphical representation of the data


    """

    def __init__(self):
        super().__init__()
        self.properties = SinglePlot1DProperties()
        self.data = None

    @staticmethod
    def applicable(dataset):
        return hasattr(dataset.data, 'residual') \
               and dataset.data.data.ndim == 1

    def _create_plot(self):
        plot_function = getattr(self.axes, self.type)
        self.data, = plot_function(self.dataset.data.axes[0].values,
                                   self.dataset.data.residual +
                                   self.dataset.data.data,
                                   label=self.properties.data.label)
        if not self.properties.drawing.label:
            self.properties.drawing.label = 'fit'
        super()._create_plot()


class SinglePlot1DProperties(aspecd.plotting.SinglePlot1DProperties):
    """
    Properties of a 1D single plot, defining its appearance.

    Additionally to the properties of the superclass, properties
    particularly for displaying both, data and fitted model exist.

    Attributes
    ----------
    data : :class:`aspecd.plotting.LineProperties`
        Properties of the line representing the data.

        "Data" here refers to the data the model has been fitted to.

        For the properties that can be set this way, see the documentation
        of the :class:`aspecd.plotting.LineProperties` class.

    """

    def __init__(self):
        super().__init__()
        self.data = aspecd.plotting.LineProperties()
        self.data.label = 'data'
        self.data.color = '#666'

    def apply(self, plotter=None):
        super().apply(plotter=plotter)
        if plotter.data:
            self.data.apply(drawing=plotter.data)
