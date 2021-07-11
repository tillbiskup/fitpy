========
Concepts
========

FitPy is built around a small set of general concepts, each aiming at complete reproducibility---and whereever possible replicability---of data acquisition, processing, and analysis. Despite its origin in spectroscopy, FitPy is generally agnostic with respect to the data processed.

Due to being based on the `ASpecD framework for reproducible data analysis <https://www.aspecd.de/>`_, FitPy uses many of the underlying concepts, namely the dataset, model, and for the actual fitting the analysis step. Furthermore, recipe-driven data analysis is used as a user interface making it possible to reliably and reproducibly fit data with no programming skills needed.


Dataset
=======

*Unit of data and metadata, prerequisite for a semantic understanding within the routines.*

Every measurement (or calculation) produces (raw) data that are useless without additional information, such as experimental parameters. This additional information is termed "metadata" within the FitPy framework. A dataset is the unit of (numerical) data and metadata. Another integral aspect is the history containing all relevant information regarding each single processing step performed on the data of the dataset.

Additionally to combining numerical data and metadata, a dataset provides a common structure, unifying the different file formats used as source for both, data and metadata. Hence, the actual data format does not matter, greatly facilitating dealing with data from different sources (and even different kinds of data).


Model
=====

*Mathematical models depending on parameters that get fitted to the data.*

Models can come from different sources. The ASpecD framework ships with a list of rather basic mathematical models that can be combined in arbitrary ways. This allows already for fitting a series of Lorentzians or Gaussians to spectroscopic data.

However, FitPy accepts every model as long as it behaves like a model of the ASpecD framework. Therefore, packages derived from the ASpecD framework and focussing on distinct spectroscopic methods may come with their own models. Similarly, you can write packages focussing only on providing models (*i.e.* simulations) of spectroscopic data of a particular kind and plug these into FitPy.


Fitting
=======

*Fitting models to data by algorithmically adjusting the parameters of the former.*

A fitting is actually an analysis step in terms of the ASpecD framework.


.. _tasks:

Recipe-driven data analysis
===========================

*Reproducible and reliable data analysis with no programming skills needed.*

Processing data consists of lots of different single tasks that can mostly be automated. This is the idea behind recipe-driven data analysis: lists of datasets and tasks that can easily be created by a user and processed fully automated. "Tasks" has a broad meaning here, including basically every automatable aspect of data analysis, including processing and analysis steps, creating representations and annotations, and finally reports.

Recipe-driven data analysis is carried out fully unattended (non-interactive). This allows to use it in context of separate hardware and a scheduling system. Situations particularly benefiting from this approach are either many datasets that need to be processed all in the same way, or few datasets requiring expensive processing such as simulation and fitting. The latter is even more true in context of global fitting and/or sampling of different starting parameters, such as Monte-Carlo or Latin-Hypercube sampling approaches.
