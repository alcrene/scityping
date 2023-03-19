List of supported types
=======================

Base types
----------

.. currentmodule:: scityping

.. autosummary::
   :nosignatures:

   NoneType
   Complex
   Range
   Slice
   Number
   Integral
   Real
   Type

Functions
---------

.. currentmodule:: scityping.functions

.. autosummary::
   :nosignatures:

   PureFunction

A function is termed `“pure” <https://en.wikipedia.org/wiki/Pure_function>`_ if its output depends only on its inputs. If you serialize a function as part of a reproducible workflow, this is generally what you want.


NumPy types
-----------

.. currentmodule:: scityping.numpy

.. autosummary::
   :nosignatures:

   DType
   NPValue
   Array
   NPGenerator
   RandomState

SciPy distributions
-------------------

.. currentmodule:: scityping.scipy

.. autosummary::
   :nosignatures:

   Distribution
   UniDistribution
   MvDistribution
   MvNormalDistribution

Univariate distributions in `scipy.stats <https://docs.scipy.org/doc/scipy/tutorial/stats.html>`_ are implemented in a generic manner, which allows us to support all of them with the same type annotation `UniDistribution`.
This is different for multivariate, which each require their own type.

`MvDistribution` is an abstract type for all multivariate distributions, and `Distribution` an abstract type for all univariate and multivariate ones.

Units
-----

Pint
^^^^

.. currentmodule:: scityping.pint

.. autosummary::
   :nosignatures:

   PintQuantity
   PintUnit

Quantities
^^^^^^^^^^

.. currentmodule:: scityping.quantities

.. autosummary::
   :nosignatures:

   QuantitiesQuantity
   QuantitiesUnit
   QuantitiesDimension

PyTorch
-------

.. currentmodule:: scityping.torch

.. autosummary::
   :nosignatures:

   Tensor
   Module
   torch_module_state_decoder
   torch_module_state_encoder
   Generator