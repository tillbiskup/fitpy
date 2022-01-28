=================
Dataset structure
=================

The dataset is an essential :doc:`concept <concepts>` of the ASpecD framework, and hence of the FitPy package.

Developers frequently need to get an overview of the structure of the dataset as defined in the FitPy package. Whereas the API documentation of  :class:`fitpy.dataset.CalculatedDataset` and :class:`fitpy.dataset.CalculatedDatasetLHS` provides a lot of information, a simple and accessible presentation of the dataset structure is often what is needed.

Therefore, the structures of the dataset classes defined in :mod:`fitpy.dataset` are provided below in YAML format, automatically generated from the actual source code.


Calculated dataset
==================

class: :class:`fitpy.dataset.CalculatedDataset`

.. literalinclude:: CalculatedDataset.yaml
   :language: yaml



Calculated dataset LHS
======================

class: :class:`fitpy.dataset.CalculatedDatasetLHS`

.. literalinclude:: CalculatedDatasetLHS.yaml
   :language: yaml

