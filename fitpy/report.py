"""Well-formatted presentation of the results of fitting models to datasets.

Fitting itself is a quite complicated process, and it is crucial for routine
use to have the results presented in automatically generated reports that
are well-formatted presentations of the results of fitting a model to a
dataset. Key to a useful report is a uniform formatting with the same
information always on the same place, allowing for easily comparing
different fits.

"""

import aspecd.report


class LaTeXFitReporter(aspecd.report.LaTeXReporter):

    def __init__(self):
        super().__init__()
        self.package = 'fitpy'
