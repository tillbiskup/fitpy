"""Well-formatted presentation of the results of fitting models to datasets.

Fitting itself is a quite complicated process, and it is crucial for routine
use to have the results presented in automatically generated reports that
are well-formatted presentations of the results of fitting a model to a
dataset. Key to a useful report is a uniform formatting with the same
information always on the same place, allowing for easily comparing
different fits.

"""
import copy
import os.path

import aspecd.plotting
import aspecd.report

import fitpy.analysis
import fitpy.plotting


class LaTeXFitReporter(aspecd.report.LaTeXReporter):
    """LaTeX Reporter for fit results.

    In addition to the functionality provided by its superclass,
    this reporter automatically creates an overview figure based on the
    dataset it gets supplied. The figure is saved using a generic name
    derived from the filename of the rendered report.


    Examples
    --------
    For convenience, a series of examples in recipe style (for details of
    the recipe-driven data analysis, see :mod:`aspecd.tasks`) is given below
    for how to make use of this class. The examples focus each on a single
    aspect.

    .. note::

        Usually, you will have set another ASpecD-derived package as
        default package in your recipe for processing and analysing your data.
        Hence, you need to provide the package name (fitpy) in the ``kind``
        property, as shown in the examples.


    In its simplest form, you just define a template and a filename for the
    resulting report.

    .. code-block:: yaml

        - kind: fitpy.report
          type: LaTeXFitReporter
          properties:
            template: simplefit.tex
            filename: test_report.tex
          compile: true

    Here, we make use of the ``simplefit.tex`` template. The results will be
    stored in ``test_report.tex`` and the LaTeX file will be compiled into a
    PDF document automatically (setting ``compile`` to true).

    """

    def __init__(self):
        super().__init__()
        self.package = 'fitpy'
        self.dataset = None

    def _render(self):
        if self.dataset and self.filename:
            self._create_figure()
        super()._render()

    def _create_figure(self):
        dataset = copy.deepcopy(self.dataset)
        plotter = fitpy.plotting.SinglePlotter1D()
        plotter.parameters['tight_layout'] = True
        plotter.parameters['tight'] = 'x'
        plotter.parameters['show_legend'] = True
        basename, _ = os.path.splitext(self.filename)
        figure_filename = "".join([basename, '-fig', '.pdf'])
        saver = aspecd.plotting.Saver()
        saver.filename = figure_filename
        plot = dataset.plot(plotter)
        plot.save(saver)

        self.context['figure_filename'] = figure_filename
        self.includes.append(figure_filename)


class LaTeXLHSFitReporter(LaTeXFitReporter):
    """LaTeX Reporter for fit results using LHS.

    In addition to the functionality provided by its superclass,
    this reporter automatically extracts the statistics from the dataset
    supplied and creates a figure showing the robustness of the LHS approach.
    The figure is saved using a generic name derived from the filename of
    the rendered report.


    Examples
    --------
    For convenience, a series of examples in recipe style (for details of
    the recipe-driven data analysis, see :mod:`aspecd.tasks`) is given below
    for how to make use of this class. The examples focus each on a single
    aspect.

    .. note::

        Usually, you will have set another ASpecD-derived package as
        default package in your recipe for processing and analysing your data.
        Hence, you need to provide the package name (fitpy) in the ``kind``
        property, as shown in the examples.


    In its simplest form, you just define a template and a filename for the
    resulting report.

    .. code-block:: yaml

        - kind: fitpy.report
          type: LaTeXLHSFitReporter
          properties:
            template: lhsfit.tex
            filename: test_report.tex
          compile: true

    Here, we make use of the ``lhsfit.tex`` template. The results will be
    stored in ``test_report.tex`` and the LaTeX file will be compiled into a
    PDF document automatically (setting ``compile`` to true).

    """

    def _create_figure(self):
        super()._create_figure()
        dataset = copy.deepcopy(self.dataset)
        analysis = fitpy.analysis.ExtractLHSStatistics()
        analysis = dataset.analyse(analysis)
        plotter = aspecd.plotting.SinglePlotter1D()
        plotter.parameters['tight_layout'] = True
        plotter.properties.drawing.linestyle = ''
        plotter.properties.drawing.marker = 'o'
        basename, _ = os.path.splitext(self.filename)
        figure_filename = "".join([basename, '-lhsfig', '.pdf'])
        saver = aspecd.plotting.Saver()
        saver.filename = figure_filename
        plot = analysis.result.plot(plotter)
        plot.save(saver)

        self.context['lhs_figure_filename'] = figure_filename
        self.includes.append(figure_filename)
