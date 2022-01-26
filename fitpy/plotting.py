"""
Graphical presentation of the results of fitting models to datasets.

Fitting itself is a quite complicated process, and it is crucial for routine
use to have graphical representations of the results, besides
automatically generated reports (see the :mod:`reports` module) that usually
contain such graphical representations.

Typically, graphical representations of fit results will contain both,
the original data and the fitted model, at least as long as datasets with
1D data are concerned. Additionally, one may be interested in plotting the
residual as well.

Technically, the plotters in the FitPy package rely on the additional
properties of the :class:`fitpy.dataset.CalculatedDataset` class,
particularly the property ``residual`` within the ``data`` property.
Therein the residual of the fit is stored, and thus, the original data can
simply be recovered by adding the residual to the fitted model contained
in the data of the dataset.


Types of plots
==============

Generally, at least two types of plotters can be distinguished with
respect to the kind of information that should be represented graphically:

* Graphical representation of both, data and fitted model

  The simplest type of such a plotter displays both, data and fitted model
  in one axis, perhaps with the residual in a second, smaller axes
  underneath.

  For 2D datasets, things become more complicated, but here, at least
  fitted model and residual can be plotted in two axes.

* Graphical representation of the reliability and quality of fits

  Particularly for robust fits including LHS or similar sampling methods,
  a graphical representation of the quality of the fit (*e.g.*,
  the fitness value plotted as function of the index of the sample) is of
  great value.


Module documentation
====================

"""

import aspecd.plotting


class SinglePlotter1D(aspecd.plotting.SinglePlotter1D):
    # noinspection PyUnresolvedReferences
    """1D plots of single datasets, including original data and fitted model.

    Convenience class taking care of 1D plots of single datasets. The type
    of plot can be set in its :attr:`aspecd.plotting.SinglePlotter1D.type`
    attribute. Allowed types are stored in the
    :attr:`aspecd.plotting.SinglePlotter1D.allowed_types` attribute.

    Additionally to the functionality of the superclass, this plotter
    displays the (experimental) data together with the fitted model.


    Attributes
    ----------
    properties : :class:`SinglePlot1DProperties`
        Properties of the plot, defining its appearance

        For the properties that can be set this way, see the documentation
        of the :class:`fitpy.plotting.SinglePlot1DProperties` and
        :class:`aspecd.plotting.SinglePlot1DProperties` classes.

    data : :class:`matplotlib.artist.Artist`
        Actual graphical representation of the data


    Examples
    --------
    For convenience, a series of examples in recipe style (for details of
    the recipe-driven data analysis, see :mod:`aspecd.tasks`) is given below
    for how to make use of this class. Of course, all parameters settable
    for the superclasses can be set as well. The examples focus each on a
    single aspect.

    .. note::

        Usually, you will have set another ASpecD-derived package as
        default package in your recipe for processing and analysing your data.
        Hence, you need to provide the package name (fitpy) in the ``kind``
        property, as shown in the examples.


    In the simplest case, just invoke the plotter with default values:

    .. code-block:: yaml

       - kind: fitpy.singleplot
         type: SinglePlotter1D
         properties:
           filename: output.pdf

    """

    def __init__(self):
        super().__init__()
        self.properties = SinglePlot1DProperties()
        self.data = None

    @staticmethod
    def applicable(dataset):
        """Check whether plot is applicable to the given dataset.

        Returns
        -------
        applicable : :class:`bool`
            `True` if successful, `False` otherwise.

        """
        return hasattr(dataset.data, 'residual') and dataset.data.data.ndim == 1

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

    Furthermore, some sensible settings for both, data and fitted model,
    are provided, such as labels and line colours. Of course, you can
    override these values manually.

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
        self.data.color = '#999'

    def apply(self, plotter=None):
        """
        Apply properties to plot.

        Parameters
        ----------
        plotter: :class:`SinglePlotter1D`
            Plotter the properties should be applied to.

        """
        super().apply(plotter=plotter)
        if plotter.data:
            self.data.apply(drawing=plotter.data)
