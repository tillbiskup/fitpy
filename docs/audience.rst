===============
Target audience
===============

Who is the target audience of the FitPy framework? Is it interesting for me?


Spectroscopists aiming at reproducibility
=========================================

The FitPy framework addresses every scientist working with data (both, measured and calculated) on a daily base. As FitPy is developed by an (experimental) spectroscopist, the focus is mainly on spectroscopy for the time being, although the underlying concepts should be sufficiently general to be useful in other context as well.

The aim of the FitPy framework is to allow the scientist to focus on the actual science (here: how to fit models to data) rather than on programming. Probably no spectroscopist can avoid being exposed to some form of computer programming. However, scripting the optimisation of a model to a given dataset is one thing, doing it reproducibly and in a way another person can easily understand what happened years later an entire different story. This is where FitPy comes in, providing powerful abstractions (recipe-driven data analysis) and ensuring full reproducibility at no additional cost.


Motivation and general ideas
============================

The motivation and general idea behind the FitPy framework is to ensure **reproducibility** and---as much as possible---replicability of data processing, starting from recording data and ending with their final (graphical) representation, e.g., in a peer-reviewed publication. This is done by mostly automatically creating a gap-less record of each processing step.

Every scientist should be well familiar with the concept of reproducibility and its importance for science in general. If you don't really care about reproducibility, you will probably have hard time using the FitPy framework. Particularly those aspects of the framework ensuring reproducibility may impair your personal freedom of doing things your way. If, however, you do care about reproducibility and are looking for a system that helps you to achieve this goal, FitPy may well be interesting for you.


Capabilities and limits
=======================

FitPy aims at providing scientists with easy ways to employ complex and state-of-the-art fitting strategies to their data. All actual fitting is done by the great packages FitPy relies on, namely the `SciPy software stack <https://www.scipy.org/>`_ and `lmfit <https://lmfit.github.io/lmfit-py/>`_. FitPy adds reproducibility and a unified user interface, *i.e.* recipe-driven data analysis, thanks to being based on the `ASpecD framework <https://docs.aspecd.de/>`_.

FitPy, however, does *not* tell you how to fit your data and which strategies to apply, nor how to properly analyse the results of your fits. This is and remains solely your responsibility as a scientist. On the other hand, FitPy allows you and others to track exactly what you have done. Therefore, it ideally contributes to the overall quality of science, allowing others to access and judge the accuracy, sensibility, and quality of published work.
