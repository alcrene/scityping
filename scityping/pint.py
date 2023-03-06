"""
Pydantic-aware classes for Pint objects.

In most cases, this should use the correct unit registry automatically.
If that is not the case, the registry can be set by assigning it as a
*class* attribute to `PintUnit`:
>>> import pint
>>> ureg = pint.UnitRegistry()
>>>
>>> from scityping.pint import PintUnit
>>> PintUnit.ureg = ureg
"""

import abc
from typing import Union, Any, Tuple
import pint

from .base import ABCSerializable, Serializable
from .base_types import SerializedData, Number

UnitRegistry = pint.UnitRegistry
ApplicationRegistry = getattr(pint, 'ApplicationRegistry',
                              getattr(pint, 'LazyRegistry', None))
# NB: Pint v0.18 changed to 'ApplicationRegistry'; before it was 'LazyRegistry'.
#     The second `getattr` is in case LazyRegistry is removed in the future

class PintQuantity(Serializable, pint.Quantity):  # Using the same name => a serializers for pint.Quantity is automatically added
    """
    A value with pint units, e.g. `1*ureg.s`.
    """
    class Data(SerializedData):
        data: Tuple[Any, Tuple[Tuple[str, Number],...]]  # Value, units
        def encode(v): return {'data': v.to_tuple()}
        def decode(data): return PintUnit.ureg.Quantity.from_tuple(data.data)

PintValue = PintQuantity  # Deprecated; for backwards compatibility
Quantity = PintQuantity   # To match the name in Pint

type_to_register = pint.Quantity
cls = PintQuantity
ABCSerializable.register(type_to_register)
ABCSerializable._base_types[cls] = type_to_register
for C in cls.mro():
    if issubclass(C, Serializable):
        C._registry[type_to_register] = cls

class PintUnitMeta(abc.ABCMeta):  # Inherit from ABCMeta because PintUnit subclasses Serializable

    @property
    def ureg(cls) -> Union[UnitRegistry, ApplicationRegistry]:
        return cls._ureg or cls._get_and_set_application_registry()

    @ureg.setter
    def ureg(cls, value: Union[UnitRegistry, ApplicationRegistry]):
        if not isinstance(value, (UnitRegistry, ApplicationRegistry)):
            raise TypeError(f"The registry assigned to `{cls.__name__}.ureg` must be a "
                            f"Pint UnitRegistry. Received {value} (type: {type(value)}).")
        cls._ureg = value
    def _get_and_set_application_registry(cls):
        cls._ureg = pint.get_application_registry()
        return cls._ureg

class PintUnit(Serializable, pint.Unit, metaclass=PintUnitMeta):
    # TODO: Type check assignment to ureg
    _ureg: UnitRegistry = None

    def apply(self, value) -> pint.Quantity:
        """
        Give `value` these units. In contrast to using multiplication by a unit
        to construct a `~Pint.Quantity`, `apply` is meant for situations where
        we are unsure whether `value` already has the desired units.

        - If `value` has no units: equivalent to multiplying by `self`.
        - If `value` has the same units as `self`: return `value` unchanged.
        - If `value` has different units to `self`: raise `ValueError`.

        Perhaps a more natural, but more verbose, function name would be
        "ensure_units".
        """
        if (not isinstance(value, pint.Quantity)
              or value.units == PintUnit.ureg.dimensionless):
            return value * self
        elif value.units == self:
            return value
        else:
            try:
                return value.to(self)
            except pint.DimensionalityError as e:
                raise ValueError(f"Cannot apply units `{self}` to value `{value}`: "
                                 f"it already has units `{value.units}`.") from e

    class Data(SerializedData):
        name: str
        def encode(unit): return str(unit)
        def decode(data): return PintUnit.ureg.Unit(data.name)

    # Allow converting a magnitude 1 Quantity to a Unit
    @classmethod
    def validate(cls, v, field=None):
        if isinstance(v, pint.Quantity):
            if v.magnitude != 1:
                raise ValueError("Quantities can only be converted to units "
                                 "if they have unit magnitude.")
            return v.units
        else:
            return super().validate(v, field=field)

type_to_register = pint.Unit
cls = PintUnit
ABCSerializable.register(type_to_register)
ABCSerializable._base_types[cls] = type_to_register
for C in cls.mro():
    if issubclass(C, Serializable):
        C._registry[cls] = cls
        C._registry[type_to_register] = cls
