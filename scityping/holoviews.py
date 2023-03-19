from __future__ import annotations

import logging
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import fields

import holoviews as hv
from .base import Serializable
from .base_types import SerializedData, Number, Type
from .functions import PureFunction
from .typing import StrictStr, StrictInt, StrictFloat

logger = logging.getLogger(__name__)

# Until we support serializing generic parameters, these are the only Parameter
# values that can be serialized
Parameter = Union[StrictStr, StrictInt, StrictFloat]

class Dimension(hv.Dimension, Serializable):
    class Data(SerializedData):
        name        : str
        label       : str
        cyclic      : bool
        default     : Optional[Parameter]
        nodata      : Optional[int]
        range       : Tuple[Union[None,Number], Union[None,Number]]
        soft_range  : Tuple[Union[None,Number], Union[None,Number]]
        step        : Optional[Number]
        type        : Optional[Type]
        unit        : Optional[str]
        value_format: Optional[PureFunction]
        values      : List[Parameter]

        def encode(dim: hv.Dimension) -> Dimension.Data:
            default = dim.default
            values = dim.values
            explain = ("at present a generic serializer for holoviews "
                       "Parameters is not implemented, so the only defaults "
                       f"of type {Parameter.__args__} are serialized.")
            if default is not None and not isinstance(default, Parameter.__args__):
                logger.warning(f"The default value {default} of dimension '{dim.label}' "
                               f"will not be serialized: {explain}")
                default = None
            if dim.values is not None and any(not isinstance(v, Parameter.__args__) for v in dim.values):
                logger.warning("Some or all of the `values` attribute of dimension "
                               f"'{dim.label}' will not be serialized: {explain}")
                values = [v for v in values if isinstance(v, Parameter.__args__)]
                if len(values) == 0:
                    values = None

            return dict(name=dim.name, label=dim.label, cyclic=dim.cyclic,
                        default=default, nodata=dim.nodata, range=dim.range,
                        soft_range=dim.soft_range, step=dim.step, type=type,
                        unit=dim.unit, value_format=dim.value_format,
                        values=dim.values)

        def decode(data: Dimension.Data) -> hv.Dimension:
            # NB: 'name' must be passed as positional arg
            kwargs = {k: getattr(data, k)
                      for k in fields(data) if k != "name"}
            return hv.Dimension(data.name, **kwargs)

