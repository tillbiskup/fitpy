FitPy
=====

FitPy is a **framework** for the **advanced fitting of models to spectroscopic data** focussing on **reproducibility**. Supported are semi-stochastic sampling of starting conditions, global fitting of several datasets at once, and fitting several concurrent models to one dataset. FitPy builds upon and extends the `ASpecD framework <https://www.aspecd.de/>`_. At the same time, it relies on the `SciPy software stack <https://www.scipy.org/>`_ and on `lmfit <https://lmfit.github.io/lmfit-py/>`_ for its fitting capabilities.

Making use of the concept of **recipe-driven data analysis**, actual fitting **no longer requires programming skills**, but is as simple as writing a text file defining both, the model and the fitting parameters in an organised way. Curious? Have a look at the following example::

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


For more general information on the FitPy framework see its `homepage <https://www.fitpy.de/>`_, and for how to use it, its `documentation <https://docs.fitpy.de/>`_.


Features
--------

A list of features, planned for the first public release:

* Framework for the advanced fitting of models to spectroscopic data focussing on reproducibility.

* Simple user interface requiring no programming skills.

* Semi-stochastic sampling of starting conditions (Latin hypercube sampling, LHS)

* Global fitting of several datasets at once

* Fitting of several concurrent models (*i.e.*, "species") to one dataset


.. warning::
  FitPy is currently under active development and still considered in Alpha development state. Therefore, expect frequent changes in features and public APIs that may break your own code. Nevertheless, feedback as well as feature requests are highly welcome.


Installation
------------

Install the package by running::

    pip install fitpy


Related projects
----------------

There is a number of related packages that are based on the ASpecD framework and each focus on one particular type of spectroscopy. The most mature packages available to date are:

* `ASpecD <https://docs.aspecd.de/>`_

  A Python framework for the analysis of spectroscopic data focussing on reproducibility and good scientific practice. The framework the cwepr package is based on, developed by T. Biskup.

* `trepr <https://docs.trepr.de/>`_

  Package for processing and analysing time-resolved electron paramagnetic resonance (TREPR) data, developed by J. Popp and maintained by T. Biskup.

* `cwepr <https://docs.cwepr.de/>`_

  Package for processing and analysing continuous-wave electron paramagnetic resonance (cw-EPR) data, originally implemented by P. Kirchner, currently developed and maintained by M. Schröder and T. Biskup.

You may as well be interested in the `LabInform project <https://www.labinform.de/>`_ focussing on the necessary more global infrastructure in a laboratory/scientific workgroup interested in more `reproducible research <https://www.reproducible-research.de/>`_. In short, LabInform is "The Open-Source Laboratory Information System".

Finally, don't forget to check out the website on `reproducible research <https://www.reproducible-research.de/>`_ covering in more general terms aspects of reproducible research and good scientific practice.


License
-------

This program is free software: you can redistribute it and/or modify it under the terms of the **BSD License**.
