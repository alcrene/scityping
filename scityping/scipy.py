"""
Manifest
========

Types:
------
  + Distributions
    - UniDistribution
    - MvDistribution (deserialization only)
    - MvNormalDistribution

JSON encoders
-------------
  + stats
    - scipy.stats._distn_infrastructure.rv_frozen
    - scipy.stats._multivariate.multi_rv


Extending support to new distribution types
===========================================

For distributions defined in scipy.stats, the best place to add support for
them is in this module. However it is also possible for external packages to
extend the `Distribution` hierarchy; this can be useful in at least two
situations:
- To support custom subclasses of scipy distributions;
- To add support for a missing scipy distribution type, without having to
  maintain one's own version of `scityping`.

The main additional consideration when extending these classes in a separate
module is to ensure that all types used in annotations here are also imported
in the module providing the extension. Otherwise Pydantic will complain about
unprepared types still being a `ForwardRef`.

    from scipy import stats
    from pydantic.dataclasses import dataclass

    from scityping.scipy import RVArg, RNGState   # Required to resolve forward refs
    from scityping.scipy import MvDistribution

    MvT = stats._multivariate.multivariate_t_frozen

    class MultivariateTDistribution(MvDistribution, MvT):
        @dataclass
        class Data(MvDistribution.Data):
            @staticmethod
            def encode(rv, include_rng_state=True):
                ...

"""

# ####
# SciPy statistical distributions
from __future__ import annotations

import abc
import logging

import numpy as np
from scipy import stats

from .utils import ModuleList
from .base import json_like, Serializable, ABCSerializable
from .base_types import SerializedData

from typing import Union, Any, Tuple, List, Dict
from .numpy import Array, NPGenerator, RandomState

logger = logging.getLogger(__name__)

# Define types used in Data objects, so classes wanting to extend this know what to import
RVArg = Union[Array[np.number], float, int, Any, None]  # Union of all accepted types for distribution arguments
RNGState = Union[None, NPGenerator, RandomState]
# FIXME: Is there a public name for frozen data types ?
RVFrozen = stats._distn_infrastructure.rv_frozen
MvRVFrozen = stats._multivariate.multi_rv_frozen
MvNormalFrozen = stats._multivariate.multivariate_normal_frozen
# List of modules searched for distribution names; precedence is given to
# modules earlier in the list
# Note that modules can also be specified as strings (so "scipy.stats" would
# also work below). This allows modules to add themselves to this list.
stat_modules = ModuleList([stats])

## Generic definitions ##

class Distribution(Serializable):
    """
    Pydantic-aware type for SciPy _frozen_ distributions. A frozen distribution
    is one for which the parameters (like `loc` and `scale`) are fixed.

    Note that any subclass of `Distribution` must also have its nested `Data`
    be a subclass of `Distribution`. This guarantee facilitates the implementation
    of some decoders, by allowing things like ``isinstance(data, Distribution.Data)``.
    (We use this for example in our Mixture implementation.)
    """
    class Data(abc.ABC, SerializedData):
        # Some custom distribution types (e.g. mixture) may have distributions,
        # or lists of distributions, as arguments.
        # We accommodate this by including them in the type.
        # NOTE: It is still better to use separate fields for Distribution,
        # and especially List[Distribution] arguments, since those have
        # much more understandable error messages.
        dist: str
        args: Tuple[RVArg, ...]
        kwds: Dict[str, RVArg]
        rng_state: RNGState=None

        @staticmethod
        @abc.abstractmethod
        def valid_distname(dist_name: str) -> bool:
            """
            Return `True` if `dist_name` is a possible value for `dist` for
            this `Data` class.
            """
            return False  # Distributions must be serialized with a subclass
        @abc.abstractmethod
        def encode(rv_frozen, include_rng_state: bool=True):  # Implemented by subclasses
            raise NotImplementedError
        def decode(data):
            dist = Distribution.get_dist(data.dist)
            if dist is Distribution:
                raise RuntimeError("The data seem to have been serialized with "
                                   "the generic type `Distribution`; this should not be possible.")
            elif ( isinstance(dist, Serializable)
                   and getattr(dist.Data, "decode", Distribution.Data) is not Distribution.Data ):
                # Second condition intended to prevent infinite recursion
                return dist.Data.decode(data)
            else:
                frozen_dist = dist(*data.args, **data.kwds)
                frozen_dist.random_state = data.rng_state
                return frozen_dist

        def __new__(cls, dist, *a, **kw):
            # Consider if a type annotation specifies Serialized[MvDistribution]
            # and we previde the serialized form of MvNormalDistribution
            # => We would to replace `cls` (set to MvDistribution.Data)
            #    by MvNormalDistribution.Data. This is what we do here:
            #    `data_type` is the `Data` subclass we need to use.
            data_type = Distribution.get_Data_type(dist)
            assert issubclass(data_type, cls), "If `data_type` is not a subclass of `cls`, __init__ will not be called."
            if super().__new__ is object.__new__:
                return super().__new__(data_type)
            else:
                if a:  # There are positional arguments:
                    return super().__new__(data_type, dist, *a, **kw)
                else:  # `dist` might be a keyword-only argument
                    return super().__new__(data_type, dist=dist, **kw)
        # # TODO
        # @classmethod
        # def __modify_schema__(cls, field_schema):
        #     field_schema.update(type='array',
        #                         items=[{'type': 'string'},
        #                                {'type': 'string'},  # dist name
        #                                {'type': 'array'},   # dist args
        #                                {'type': 'array'},   # dist kwds
        #                                {'type': 'array'}    # random state (optional) Accepted: int, old RandomState, new Generator
        #                                ])

    def __init_subclass__(cls):
        data_cls = getattr(cls, "Data", None)
        supercls = next(filter(None,
            (base if issubclass(base, Distribution) else False for base in cls.mro()[1:])))  # The `.Data` attribute of the nearest parent
        if  (  data_cls is None
               or not isinstance(data_cls, type)
               or not issubclass(data_cls, supercls.Data) ):
            base_name = f"{supercls.__module__}.{supercls.__qualname__}"
            raise TypeError(f"{cls.__name__} is a `{base_name}` "
                            "subclass and must define a nested `Data` class "
                            f"as a subclass of `{base_name}.Data`.")
        super().__init_subclass__()

    # The distinction that only frozen dists are serializable is not obvious,
    # so we wrap `validate` to catch that error and print an explanation message
    @classmethod
    def validate(cls, v):
        if isinstance(v, (stats._distn_infrastructure.rv_generic,
                          stats._multivariate.multi_rv_generic)):
            raise TypeError("`Distribution` expects a frozen random variable; "
                            f"received '{v}' with unspecified parameters.")
        return super().validate(v)

    @staticmethod
    def get_dist(name: str):
        """Search the `stat_modules` for a distribution type matching `name`."""
        for module in stat_modules:
            dist = getattr(module, name, None)
            if dist:
                break
        if dist is None:
            raise RuntimeError("Unable to find a distribution named "
                               f"'{name}' within the modules {stat_modules}. "
                               "To add a module to search for distributions, "
                               f"update the list at {__name__}.stat_modules.")
        return dist

    @staticmethod
    def get_Data_type(dist_name: str):
        """
        From the `dist_name` included in serialized data, determine the subclass
        of `Distribution.Data` which produced it.
        """
        data_types = tuple(set(dist_type.Data
                               for dist_type in Distribution._registry.values()
                               if dist_type.Data.valid_distname(dist_name)))
        # Remove types which are strict subclasses of another
        to_remove = set()
        for i, T in enumerate(data_types):
            if issubclass(T, data_types[:i] + data_types[i+1:]):
                to_remove.add(i)
        data_types = tuple(T for i, T in enumerate(data_types) if i not in to_remove)
        # In case of ambiguity, raise error and abort => better fix this error early than let it seep into the code
        if len(data_types) == 0:
            raise RuntimeError("No subclass of `Distribution.Data` seems to recognize "
                               f"the distribution type '{dist_name}'.\nData classes "
                               f"checked: {T.Data for T in Distribution._registry.values()}")
        elif len(data_types) > 1:
            raise RuntimeError(f"Multiple `Data` classes recognize '{dist_name}': "
                               f"{data_types}")
        else:
            return data_types[0]

## Concrete subclasses ##

class UniDistribution(Distribution, RVFrozen):
    class Data(Distribution.Data):
        @staticmethod
        def valid_distname(dist_name: str) -> bool:
            univariate_names = (o.name for o in stats.__dict__.values()
                                if isinstance(o, stats._distn_infrastructure.rv_generic))
            return dist_name in univariate_names
        def encode(rv, include_rng_state=True):
            if rv.args:
                logger.warning(
                    "For the most consistent and reliable serialization of "
                    "distributions, consider specifying them using only keyword "
                    f"parameters. Received for distribution {rv.dist.name}:\n"
                    f"Positional args: {rv.args}\nKeyword args: {rv.kwds}")
            random_state = rv.dist._random_state if include_rng_state else None
            return rv.dist.name, rv.args, rv.kwds, random_state
# NB: We need to special case multivariate distributions, because they follow
#     a different convention (and don't have a standard 'kwds' attribute)
class MvDistribution(Distribution, MvRVFrozen):
    class Data(Distribution.Data):
        @staticmethod
        def valid_distname(dist_name: str) -> bool:
            # Multivariate distributions must always be serialized with subclasses
            return False
        def encode(rv, include_rng_state=True):
            raise NotImplementedError(
                "The json_encoder for `Distribution` needs to be special "
                "cased for each multivariate distribution, and this has "
                f"not yet been done for '{rv._dist}'.")
class MvNormalDistribution(MvDistribution, MvNormalFrozen):
    class Data(MvDistribution.Data):
        @staticmethod
        def valid_distname(dist_name: str) -> bool:
            return dist_name == "multivariate_normal"
        def encode(rv, include_rng_state=True):
            dist = rv._dist
            random_state = dist._random_state if include_rng_state else None
            # name, args, kwds, random_state
            return ("multivariate_normal", (),
                    {'mean':rv.mean, 'cov':rv.cov}, random_state)

UniDistribution.register(RVFrozen)
MvDistribution.register(MvRVFrozen)  # This is only to provide the NotImplementedError message
MvNormalDistribution.register(MvNormalFrozen)
