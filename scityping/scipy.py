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
"""

# ####
# SciPy statistical distributions
from __future__ import annotations

import abc
import logging

from scipy import stats

from .base import json_like, Serializable, ABCSerializable
from .base_types import SerializedData

from typing import Union, Any, Tuple, List, Dict
from .numpy import NPGenerator, RandomState

logger = logging.getLogger(__name__)

# FIXME: Is there a public name for frozen data types ?
RVFrozen = stats._distn_infrastructure.rv_frozen
MvRVFrozen = stats._multivariate.multi_rv_frozen
MvNormalFrozen = stats._multivariate.multivariate_normal_frozen
# List of modules searched for distribution names; precedence is given to
# modules earlier in the list
stat_modules = [stats]

class Distribution(Serializable):
    """
    Pydantic-aware type for SciPy _frozen_ distributions. A frozen distribution
    is one for which the parameters (like `loc` and `scale`) are fixed.
    """
    class Data(abc.ABC, SerializedData):
        # Some custom distribution types (e.g. mixture) may have distributions,
        # or lists of distributions, as arguments.
        # We accommodate this by including them in the type.
        dist: str
        args: Tuple[Union[Distribution, List[Distribution], Any],...]
        kwds: Dict[str, Union[Distribution, List[Distribution], Any]]
        rng_state: Union[None, NPGenerator, RandomState]=None

        @abc.abstractmethod
        def encode(rv_frozen, include_rng_state: bool=True):  # Implemented by subclasses
            raise NotImplementedError
        def decode(data):
            dist = None
            for module in stat_modules:
                dist = getattr(module, data.dist, None)
                if dist:
                    break
            if dist is None:
                raise RuntimeError("Unable to find a distribution named "
                                   f"'{dist}' within the modules {stat_modules}. "
                                   "To add a module to search for distributions, "
                                   f"update the list at {__name__}.stat_modules.")
            if isinstance(dist, Serializable):
                return dist.Data.decode(data)
            else:
                frozen_dist = dist(*data.args, **data.kwds)
                frozen_dist.random_state = data.rng_state
                return frozen_dist

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

    # The distinction that only frozen dists are serializable is not obvious,
    # so we wrap `validate` to catch that error and print an explanation message
    @classmethod
    def validate(cls, v):
        if isinstance(v, (stats._distn_infrastructure.rv_generic,
                          stats._multivariate.multi_rv_generic)):
            raise TypeError("`Distribution` expects a frozen random variable; "
                            f"received '{v}' with unspecified parameters.")
        return super().validate(v)


class UniDistribution(Distribution, RVFrozen):
    class Data(Distribution.Data):
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
        def encode(rv, include_rng_state=True):
            raise NotImplementedError(
                "The json_encoder for `Distribution` needs to be special "
                "cased for each multivariate distribution, and this has "
                f"not yet been done for '{rv._dist}'.")
class MvNormalDistribution(MvDistribution, MvNormalFrozen):
    class Data(Distribution.Data):
        def encode(rv, include_rng_state=True):
            dist = rv._dist
            random_state = dist._random_state if include_rng_state else None
            # name, args, kwds, random_state
            return ("multivariate_normal", (),
                    {'mean':rv.mean, 'cov':rv.cov}, random_state)

UniDistribution.register(RVFrozen)
MvDistribution.register(MvRVFrozen)  # This is only to provide the NotImplementedError message
MvNormalDistribution.register(MvNormalFrozen)
