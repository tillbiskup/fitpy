.. _use_cases:

=========
Use cases
=========


.. sidebar:: Contents

    .. contents::
        :local:
        :depth: 1


This section provides a few ideas of how basic operation of the FitPy package may look like. It focusses on **recipe-driven data analysis** as its user-friendly interface that does not require the spectroscopist to know anything about programming and allows to fully focus on the actual fitting.

As a user, you write "recipes" in form of human-readable YAML files telling the application which tasks to perform on what datasets. This allows for fully unattended, automated and scheduled fitting. At the same time, it allows you to analyse the data without need for actual programming.


.. important::

    Currently, this section is used by the developers to get an idea of how to design the interface of FitPy. Therefore, different, not yet implemented scenarios are listed as recipes. Assume the interface to change frequently for now, as it is still in the initial design phase.


Fitting of single datasets
==========================


Most basic fitting
------------------

Generally, models are fitted to data of a dataset. While the datasets are loaded as usual, models are created using a ``model task``, while fitting is performed using an ``analysis task`` from the FitPy package. The latter inherits from the ASpecD analysis task and hence returns a calculated dataset with the fitted model as its data.


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


Settings for the algorithm
--------------------------

The algorithm used for fitting (the method) as well as other settings regarding the algorithm need to be controllable by the user.


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
            algorithm:
              method: leastsq
        result: fitted_gaussian


Omitting parts of the dataset
-----------------------------

Often, real data contain parts that cannot be described by a certain model, but can safely be ignored, or they contain outliers that shall not be fitted. Therefore, fitting needs to provide means to specify regions of the dataset to be ignored during fitting.


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
            algorithm:
              method: leastsq
            cut_range:
              - [5, 6]
              - [9, 10]
        result: fitted_gaussian


Robust fitting (sampling of starting conditions, LHS)
=====================================================

One crucial aspect of the FitPy package is to provide simple means to perform optimisation starting from different starting conditions via a Latin Hypercube Sampling (LHS). Here, both, the number of samples per parameter as well as the interval the starting conditions should be sampled from for each parameter need to be provided.

One problem occurring with sampling algorithms is that the result is no longer a single dataset, at least not trivially. It might still be a single dataset, but the information from the different runs needs to be available for analysis of the goodness of the eventual fit.


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
        type: LHSFit
        properties:
          model: gaussian_model
          parameters:
            fit:
              amplitude:
                lhs_range: [1, 10]
            lhs:
              points: 5
        result: fitted_gaussian


Fitting multiple species to one dataset
=======================================

Different to global fitting, where one model is fitted to several independent datasets, fitting multiple species to one dataset is nothing special from a fitting perspective, as a rather complex composite model is used in this case.

There are, however, a few minor differences with respect to the parameter definitions: As the parameters will often have the same name, as they stem from the same model, the corresponding fit parameter will get lists for initial guesses, ranges, and alike. Furthermore, the weighting for the different models of the composite model needs to be fitted as well.

Usually, as the number of parameters increases dramatically with more than one species, robust fitting shall be applied.


.. code-block:: yaml
    :linenos:

    format:
      type: ASpecD recipe
      version: '0.2'

    datasets:
      - /path/to/dataset

    tasks:
      - kind: model
        type: CompositeModel
        from_dataset: /path/to/dataset
        properties:
          models:
            - Gaussian
            - Gaussian
          parameters:
            - position: 5
            - position: 8
        output: model
        result: multiple_gaussians

      - kind: fitpy.singleanalysis
        type: MultipleSpeciesFit
        properties:
          model: multiple_gaussians
          parameters:
            fit:
              position:
                start:
                  - 5
                  - 8
                range:
                  - [3, 7]
                  - [6, 9]
              weights:
                start:
                  - 1
                range:
                  - [0.5, 2]
        result: fitted_gaussians


Global fitting
==============

Global fitting covers multiple independent datasets to which models with a joint set of parameters are fitted. This is different to multiple species fitted to one dataset.

As such, the fitting inherits from :class:`aspecd.analysis.MultiAnalysisStep`, and for each dataset a model needs to be provided, as the datasets cannot be restricted to have the same dimensions and ranges of their axes.


.. code-block:: yaml
    :linenos:

    format:
      type: ASpecD recipe
      version: '0.2'

    datasets:
      - /path/to/first/dataset
      - /path/to/second/dataset

    tasks:
      - kind: model
        type: Gaussian
        properties:
          parameters:
            position: 1.5
            width: 0.5
        from_dataset: /path/to/first/dataset
        output: model
        result: gaussian_model_1

      - kind: model
        type: Gaussian
        properties:
          parameters:
            position: 1.5
            width: 0.5
        from_dataset: /path/to/second/dataset
        result: gaussian_model_2

      - kind: fitpy.multianalysis
        type: GlobalFit
        properties:
          models:
            - gaussian_model_1
            - gaussian_model_2
          parameters:
            fit:
              amplitude:
                start: 5
                range: [3, 7]
        result: fitted_gaussian


Questions to address:

  * How to deal with constraints for parameters for the multiple datasets?

    Example: Data have been recorded in an angular-dependent fashion, and while the angle offset between datasets is known with some accuracy, the initial offset shall be fitted.

    In such case, one probably would want to provide the offsets, let the fitting adjust the offsets within a given range, and let the initial offset to be varied in a much wider range.



Graphical visualisation of fit results
======================================

Graphical visualisation of fit results is of crucial importance. The lmfit package provides straightforward and compelling means for most standard situations, and these can be used to inspire similar solutions based on the functionality provided by the ASpecD framework.


Comparing data and fitted model
-------------------------------

Basically, data, model, and perhaps the residual should be shown.

As the results of a fit are not contained in the original experimental dataset, but rather in a calculated dataset that is returned by the fitting step, the plotters need probably access to both, the original dataset and the fitted model residing in the calculated dataset. One could try to overcome this problem by providing the original data in some way in the calculated dataset that results from the fitting process.


Robustness of sampling strategies
---------------------------------

When sampling starting conditions, it is important to graphically display the results for the different samples, to evaluate the robustness of the fit and the applicability of the grid used.


Fit reports
===========

The importance of sensible reports cannot be overrated, and TSim is the key to the success of much of the own research, allowing a skilled student with few hours of introduction to perform fits to data without much need of further supervision besides discussing the results together.

Thanks to the report generating capabilities of the ASpecD framework, generating reports should be straight-forward. Key here is not how to generate reports, but to provide sensible templates and, where necessary and sensible, generate the necessary information to be added to the reports.

As the results of a fit are not contained in the original experimental dataset, but rather in a calculated dataset that is returned by the fitting step, the reports need probably access to both, the original dataset and the fitted model residing in the calculated dataset. One could try to overcome this problem by providing the original data in some way in the calculated dataset that results from the fitting process.

Shall reports automatically generate certain figures if these are not provided? May be sensible, but would include functionality from plotters in reports. An alternative would be to provide recipe templates for specifying the plots that can be adapted by the user upon need.


Pipelines
=========

Inspired by packages such as sklearn, it might prove useful to be able to define entire pipelines and employ a series of fitting strategies.

The question remains: Is this a separate task, or could this reasonably be done using recipe-driven data analysis and providing well-crafted example recipes?

