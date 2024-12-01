=========
Changelog
=========

This page contains a summary of changes between the official FitPy releases. Only the biggest changes are listed here. A complete and detailed log of all changes is available through the `GitHub Repository Browser <https://github.com/tillbiskup/fitpy/commits/master>`_.


Version 0.1.2
=============

Released 2024-12-01


Fixes
-----

* Fix with metadata to_dict method


Version 0.1.1
=============

Released 2024-01-15


Fixes
-----

* Changes in :class:`fitpy.plotting.SinglePlotter1D` to make it work with ASpecD >= 0.9.0.

  **Important note:** This version of FitPy requires **ASpecD >= 0.9.1** to work.


Changes
-------

* Use Black for automatic code formatting


Version 0.1.0
=============

Released 2022-01-30

* First public release

* General implementation of the interfaces

* Dedicated plotters plotting of both, data and fitted model

* Reporting capabilities

* Settings for the fit algorithm

* Semi-stochastic sampling of starting conditions (Latin hypercube sampling, LHS)


Version 0.1.0.dev1
==================

Released 2019-09-26

* First public pre-release on PyPI
