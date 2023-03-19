"""
Pydantic-aware classes for Quantities objects.
"""

import numpy as np
import quantities as pq

from .base import Serializable, ABCSerializable
from .base_types import SerializedData
from .numpy import Array

class QuantitiesQuantity(Serializable, pq.Quantity):
    class Data(SerializedData):
        data : Array  # Names match signature of pq.Quantity
        units: str
        def encode(val): return v.magnitude, str(v.dimensionality)

Quantity = QuantitiesQuantity  # Match the name in quantities

type_to_register = pq.Quantity
cls = QuantitiesQuantity
ABCSerializable.register(type_to_register)
ABCSerializable._base_types[cls] = type_to_register
for C in cls.mro():
    if issubclass(C, Serializable):
        C._registry[cls] = cls
        C._registry[type_to_register] = cls


class QuantitiesUnit(QuantitiesQuantity):
    """
    Exactly the same as QuantitiesValue, except that we enforce the magnitude
    to be 1. In contrast to Pint, Quantities doesn't seem to have a unit type.
    """
    class Data(QuantitiesQuantity.Data):
        pass
    @classmethod
    def validate(cls, v, field=None):
        v = super().__init__(v, field=field)
        if v.magnitude != 1:
            field_name = field.name if field else "<QuantitiesUnit>"
            raise ValueError(f"Field {field_name}: Units must have "
                             "magnitude of one.")
        return v

class QuantitiesDimension(Serializable, pq.dimensionality.Dimensionality):
    class Data(SerializedData):
        name: str
        def encode(dim): return str(dim)
        def decode(data): return pq.quantity.validate_dimensionality(data.name)

    # Allow casting from Quantity
    @classmethod
    def validate(cls, v, field=None):
        if isinstance(v, pq.Quantity):
            v = v.dimensionality
        return super().validate(v, field=field)
