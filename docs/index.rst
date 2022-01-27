.. FitPy documentation master file, created by
   sphinx-quickstart on Sat Nov 24 13:04:06 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: User Manual:
   :hidden:

   audience
   introduction
   concepts
   usecases
   installing

.. toctree::
   :maxdepth: 2
   :caption: Developers:
   :hidden:

   people
   developers
   changelog
   roadmap
   dataset-structure
   api/index



FitPy documentation
====================

Welcome! This is the documentation for FitPy – a **framework** for the **advanced fitting of models to spectroscopic data** focussing on **reproducibility**. Supported are semi-stochastic sampling of starting conditions, global fitting of several datasets at once, and fitting several concurrent models to one dataset. FitPy builds upon and extends the `ASpecD framework <https://www.aspecd.de/>`_. At the same time, it relies on the `SciPy software stack <https://www.scipy.org/>`_ and on `lmfit <https://lmfit.github.io/lmfit-py/>`_ for its fitting capabilities.

Making use of the concept of **recipe-driven data analysis**, actual fitting **no longer requires programming skills**, but is as simple as writing a text file defining both, the model and the fitting parameters in an organised way. Curious? Have a look at the following example:


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


For more general information on the FitPy framework see its `Homepage <https://www.fitpy.de/>`_, and for how to use it, carry on reading. Interested in more real-live examples? Check out the :ref:`use cases section <use_cases>`.



Features
--------

A list of features, not all implemented yet, but planned for the next releases:

* Advanced fitting of models to spectroscopic data focussing on reproducibility.

* Simple user interface requiring no programming skills.

* Semi-stochastic sampling of starting conditions (Latin hypercube sampling, LHS)

* Global fitting of several datasets at once

* Fitting of several concurrent models (*i.e.*, "species") to one dataset


And to make it even more convenient for users and future-proof:

* Open source project written in Python (>= 3.7)

* Developed fully test-driven

* Extensive user and API documentation


.. warning::
  FitPy is currently under active development and still considered in Alpha development state. Therefore, expect frequent changes in features and public APIs that may break your own code. Nevertheless, feedback as well as feature requests are highly welcome.


.. _sec-how_to_cite:

How to cite
-----------

FitPy is free software. However, if you use FitPy for your own research, please cite it appropriately:

Till Biskup. FitPy (2022). `doi:10.5281/zenodo.####### <https://doi.org/10.5281/zenodo.#######>`_

To make things easier, FitPy has a `DOI <https://doi.org/10.5281/zenodo.#######>`_ provided by `Zenodo <https://zenodo.org/>`_, and you may click on the badge below to directly access the record associated with it. Note that this DOI refers to the package as such and always forwards to the most current version.



Where to start
--------------

Users new to FitPy should probably start :doc:`at the beginning <audience>`, those familiar with its :doc:`underlying concepts <concepts>` may jump straight to the section explaining how :doc:`working with the FitPy framework looks like <usecases>`, particularly in context of recipe-driven data analysis.

The :doc:`API documentation <api/index>` is the definite source of information for developers, besides having a look at the source code.


Installation
------------

To install the FitPy framework on your computer (sensibly within a Python virtual environment), open a terminal (activate your virtual environment), and type in the following:

.. code-block:: bash

    pip install fitpy

Have a look at the more detailed :doc:`installation instructions <installing>` as well.


Related projects
----------------

There is a number of related packages that are based on the ASpecD framework and each focus on one particular type of spectroscopy. The most mature packages available to date are:

* `ASpecD <https://docs.aspecd.de/>`_

  A Python framework for the analysis of spectroscopic data focussing on reproducibility and good scientific practice. The framework the FitPy package is based on, developed by T. Biskup.

* `trepr <https://docs.trepr.de/>`_

  Package for processing and analysing time-resolved electron paramagnetic resonance (TREPR) data, developed by J. Popp, currently developed and maintained by M. Schröder and T. Biskup.

* `cwepr <https://docs.cwepr.de/>`_

  Package for processing and analysing continuous-wave electron paramagnetic resonance (cw-EPR) data, originally implemented by P. Kirchner, currently developed and maintained by M. Schröder and T. Biskup.

You may as well be interested in the `LabInform project <https://www.labinform.de/>`_ focussing on the necessary more global infrastructure in a laboratory/scientific workgroup interested in more `reproducible research <https://www.reproducible-research.de/>`_. In short, LabInform is "The Open-Source Laboratory Information System".

Finally, don't forget to check out the website on `reproducible research <https://www.reproducible-research.de/>`_ covering in more general terms aspects of reproducible research and good scientific practice.


License
-------

This program is free software: you can redistribute it and/or modify it under the terms of the **BSD License**. However, if you use FitPy for your own research, please cite it appropriately. See :ref:`How to cite <sec-how_to_cite>` for details.


A note on the logo
------------------

The logo shows a Latin square, usually attributed to Leonhard Euler. In the context of statistical sampling, a Latin square consists of only one sample in each row and each column. Its *n*-dimensional generalisation, the Latin hypercube, is used to generate a near-random sample of parameter values from a multidimensional distribution in statistics, e.g., for obtaining sets of starting parameters for minimisation and fitting tasks. The logo shows a snake (obviously a Python) distributed such over a 4x4 square that it visits each row and each column only once. The copyright of the logo belongs to J. Popp.
