# -*- coding: utf-8 -*-
"""
A set of Pydantic-compatible types relevant for scientific applications.
Includes types for numbers, NumPy, and PyTorch subtypes and objects.

Manifest
========

Types:
------
  + Base class
    - Serializable
  + Types (implemented in smttask.typing)
    - Types
    - Generics
  + Builtins:
    - Range
    - Sequence  (In contrast to Pydantic's Sequence, also recognizes `Range`)
    - Slice
  + Numbers:
    - Number (validation only)
    - Integral (validation only)
    - Real (validation only, except int -> float)

JSON encoders
-------------
  + Builtins:
    - complex
    - range
    - slice
    - type

TODO: JSON schemas are not correct.
"""
# TODO: __modify_schema__ should be defined in Serializable, by using the schema of the Data

from __future__ import annotations

import abc
from types import SimpleNamespace
from collections.abc import Callable as Callable_
import typing
from typing import Union, Any, Iterable, Sequence as _Sequence, Tuple, Dict

import numbers

import logging
logger = logging.getLogger(__name__)

from .base import Serializable, ABCSerializable  # Serializable is not defined in this module to prevent import cycles

# dataclasses are used for the `Data` container associated to each type.
# Use Pydantic dataclasses if available (they provide serialization/deserialization)
# Otherwise, use builtin dataclasses
try:
    import pydantic
except ModuleNotFoundError:
    from dataclasses import dataclass
else:
    from .pydantic import dataclass

__all__ = ["SerializedData", "Complex", "Range", "Slice",
           "Number", "Integral", "Real"]

# ###############
# Helper base type for the nested `Data` classes
# This is used to reduce boilerplate (remove the need for @dataclass and `self` in arguments)

class SerializedDataMeta(abc.ABCMeta):
    def __new__(metacls, cls, bases, namespace):
        for nm, attr in namespace.items():
            if (not nm.startswith("__") and isinstance(attr, Callable_)):
                # NB: staticmethod and classmethod objects are not callable
                # Dunder methods should remain methods
                namespace[nm] = staticmethod(attr)
        obj = super().__new__(metacls, cls, bases, namespace)
        return dataclass(obj)
class SerializedData(metaclass=SerializedDataMeta):
    """
    Define a helper base class for nested Data data classes, which
    - makes every method static (avoiding the need to pass an unnecessary 'self');
    - applies the @dataclass decorator to the new class, converting it to a dataclass.
    """
    @abc.abstractmethod
    def encode(value): raise NotImplementedError

# ###############
# Support for builtin types:
# - complex
# - range
# - slice
# - type

class Complex(Serializable, complex):  # Using same name allows serializing original type
  class Data(SerializedData):
    real: float
    imag: float
    def encode(z): return z.real, z.imag
    def decode(data): return complex(*data)

@ABCSerializable.register   # NB: It is not permitted to subclass `range`
class Range(Serializable):
    class Data(SerializedData):
        start: int
        stop: Optional[int]=None
        step: Optional[int]=None
        def encode(r): return r.start, r.stop, r.step
        def decode(data): return range(*data)  # Required since we don’t subclass `range`

    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     """We need to tell pydantic how to export this type to the schema,
    #     since it doesn't know what to map the type to.
    #     """
    #     # See https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types
    #     # for the API, and https://pydantic-docs.helpmanual.io/usage/schema/
    #     # for the expected fields
    #     # TODO: Use 'items' ?
    #     field_schema.update(
    #         type="array",
    #         description="('range', START, STOP, STEP)"
    #         )

@ABCSerializable.register   # NB: It is not permitted to subclass `slice`
class Slice(Serializable):
    class Data(SerializedData):
        start: int
        stop: Optional[int]=None
        step: Optional[int]=None
        def encode(r): return r.start, r.stop, r.step
        def decode(data): return slice(*data)  # Required since we don’t subclass `slice`

    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     """We need to tell pydantic how to export this type to the schema,
    #     since it doesn't know what to map the type to.
    #     """
    #     # See https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types
    #     # for the API, and https://pydantic-docs.helpmanual.io/usage/schema/
    #     # for the expected fields
    #     # TODO: Use 'items' ?
    #     field_schema.update(
    #         type="array",
    #         description="('slice', START, STOP, [STEP])"
    #         )

# ###############
# Overrides of typing types

#  Pydantic recognizes typing.Sequence, but treats it as a shorthand for
#  Union[List, Tuple] (but with equal precedence). This means that other
#  sequence types like `range` are not recognized.
Sequence = Union[Range, _Sequence]

# ###############
# Types based on numbers module

class Number(numbers.Number):
    """
    This type does not support coercion, only validation.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        if not isinstance(value, numbers.Number):
            raise TypeError(f"Field {field.name} expects a number. "
                            f"It received {value} [type: {type(value)}].")
        return value
    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(type="number")

class Integral(numbers.Integral):
    """
    This type does not support coercion, only validation.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        if not isinstance(value, numbers.Integral):
            raise TypeError(f"Field {field.name} expects an integer. "
                            f"It received {value} [type: {type(value)}].")
        return value
    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(type="integer")

class Real(numbers.Real):
    """
    This type generally does not support coercion, only validation.
    Exception: integer values are cast with `float`.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        if not isinstance(value, numbers.Real):
            raise TypeError(f"Field {field.name} expects a real number. "
                            f"It received {value} [type: {type(value)}].")
        elif isinstance(value, numbers.Integral):
            # Convert ints to floating point
            return float(value)
        return value
    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(type="number")
