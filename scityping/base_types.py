# -*- coding: utf-8 -*-
"""
A set of Pydantic-compatible types relevant for scientific applications.
Includes types for numbers, NumPy, and PyTorch subtypes and objects.

"""
# TODO: __modify_schema__ should be defined in Serializable, by using the schema of the Data

from __future__ import annotations

import abc
import typing
import inspect
from types import SimpleNamespace
from collections.abc import Callable as Callable_
from typing import Optional, Union, Any, Iterable, List, Tuple, Dict, Literal
from dataclasses import dataclass, asdict, fields

import numbers

import logging
logger = logging.getLogger(__name__)

from .base import Serializable, json_like, Dataclass, validate_dataclass
from .utils import get_type_key
from .config import config

# dataclasses are used for the `Data` container associated to each type.
# # Use Pydantic dataclasses if available (they provide serialization/deserialization)
# # Use the smart_union=True option with Pydantic dataclasses, to avoid accidental coercion
# # Otherwise, use builtin dataclasses
# # TODO: Don’t use pydantic dataclasses. Use normal ones, and include deserialization in `SerializedData`
# try:
#     import pydantic
# except ModuleNotFoundError:
#     from dataclasses import dataclass
# else:
#     from .pydantic import dataclass as pydantic_dataclass
#     dataclass = pydantic_dataclass(config={"smart_union": True})

__all__ = ["dataclass", "SerializedData",
           "NoneType", "Complex", "Range", "Slice",
           "Number", "Integral", "Real",
           "Type", "GenericType", "PydanticGenericType"]

# ###############
# Helper base type for the nested `Data` classes
# This is used to reduce boilerplate (remove the need for @dataclass and `self` in arguments)

class SerializedDataMeta(abc.ABCMeta):
    def __new__(metacls, cls, bases, namespace):
        for nm, attr in namespace.items():
            if not nm.startswith("__") and isinstance(attr, Callable_):
                # NB: staticmethod and classmethod objects are not callable
                # Dunder methods should remain methods
                sig = inspect.signature(attr)
                if sig.parameters and next(iter(sig.parameters)) == "self":
                    # If the first argument is `self`, don’t convert to a static method
                    pass
                else:
                    namespace[nm] = staticmethod(attr)
        obj = super().__new__(metacls, cls, bases, namespace)
        return dataclass(obj)
class SerializedData(Dataclass, metaclass=SerializedDataMeta):
    """
    Define a helper base class for nested Data data classes, which
    - makes every method static (avoiding the need to pass an unnecessary 'self');
    - applies the @dataclass decorator to the new class, converting it to a dataclass.
    """
    @abc.abstractmethod
    def encode(value): raise NotImplementedError
    def __post_init__(self):
        validate_dataclass(self, inplace=True)
    # ## Pydantic compatibility ##
    # # Note that instead of defining these methods, `SerializedData` could inherit
    # # from `BaseModel`, but that would make pydantic a hard dependency
    # @classmethod
    # def __get_validators__(cls):
    #     yield cls.validate
    # @classmethod
    # def validate(cls, value, field=None):  # 'field' only present for consistency
    #     return cls(**value)

# ###############
# Special type which only accepts the value `None`; used for deactivating
# fields in a subclass

class NoneType:
    """
    A Pydantic compatible 'type' which only accepts None.
    Use case: deactivating options of a parent class by forcing them to be None
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v, field=None):
        if field is None:
            field = SimpleNamespace(name="")
        assert v is None, f"Field '{field.name}' of '{cls.__qualname__}' accepts only the value `None`."
        return None

# ###############
# Support for builtin types:
# - complex
# - range
# - slice
# - type

# TODO: Since they only contain plain data types, we could make the `Data` classes more lightweight by just subclassing `tuple`.

class Complex(Serializable, complex):  # Using same name allows serializing original type
  class Data(SerializedData):
    real: float
    imag: float
    def encode(z): return z.real, z.imag
    def decode(data): return complex(data.real, data.imag)

class Range(Serializable):  # NB: It is not permitted to subclass `range`
    class Data(SerializedData):
        start: int
        stop: Optional[int]=None
        step: Optional[int]=None
        def encode(r): return r.start, r.stop, r.step
        def decode(data): return range(data.start, data.stop, data.step)  # Required since we don’t subclass `range`

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
Range.register(range)

class Slice(Serializable):  # NB: It is not permitted to subclass `range`
    class Data(SerializedData):
        start: int
        stop: Optional[int]=None
        step: Optional[int]=None
        def encode(r): return r.start, r.stop, r.step
        def decode(data): return slice(data.start, data.stop, data.step)  # Required since we don’t subclass `slice`

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
Slice.register(slice)

# ###############
# Overrides of typing types

#  Pydantic recognizes typing.Sequence, but treats it as a shorthand for
#  Union[List, Tuple] (but with equal precedence). This means that other
#  sequence types like `range` are not recognized.
Sequence = Union[Range, typing.Sequence]

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


#####################
# Type
#####################

T = typing.TypeVar('T')
class Type(typing.Type[T], Serializable):  # NB: Serializable must come 2nd
    """
    Make types serializable; the serialization format is
        ('Type', <module name>, <type name>)
    During deserialization, it effectively executes
        from <module name> import <type name>

    .. Caution:: **Limitation** As with `typing.Type`, one can indicate the specific type between
       brackets; e.g. ``Type[int]``, and Pydantic will enforce this restriction.
       However at present deserialization only works when the type is unspecified.

    .. Warning:: This kind of serialization will never be 100% robust and
       should be used with care. In particular, since it relies on <module name>
       remaining unchanged, it is certainly not secure. (Although no less so
       than `pickle`.)
       Because of the potential security issue, it requires adding modules where
       tasks are defined to the ``scityping.config.safe_packages`` whitelist.
    """
    # FIXME: When the type T is specified, the specialized type doesn't inherit __get_validators__
    #   (although it does inherit the other methods)
    class Data(SerializedData):
        module: str
        name: str

        @classmethod
        def encode(cls, T: typing.type) -> Type.Data:
            if not isinstance(T, type):
                raise TypeError(f"'{T}' is not a type.")
            if T.__module__ == "__main__":
                # raise ValueError("Can't serialize types defined in the '__main__' module.")
                logger.warning(f"Type '{T}' was serialized in the '__main__' module; "
                               "it will only be deserializable within the same script "
                               "or notebook.")
            return cls(T.__module__, T.__qualname__)

        def decode(data: Type.Data) -> Type:
            from importlib import import_module
            module = data.module
            if (any(module.startswith(sm) for sm in config.safe_packages)
                  or config.trust_all_inputs):
                m = import_module(module)
                for _name in data.name.split("."):
                    try:
                        m = getattr(m, _name)
                    except AttributeError as e:
                        if module == "__main__":
                            raise RuntimeError(
                                f"The type '{data.name}' was serialized in the same "
                                "namespace in which it was defined; therefore it can "
                                "also only be deserialized in that namespace. "
                                "However no type of that name was found in the current "
                                "`__main__` namespace")
                        else:
                            raise e
                else:  # All dotted name elements were found
                    T = m
            else:
                raise RuntimeError(
                    "As with pickle, deserialization of types can lead to "
                    "arbitrary code execution. It is only permitted after "
                    f"adding '{module}' to ``scitying.config.safe_packages`` "
                    "(recommended) or setting the option "
                    "``scityping.config.trust_all_inputs = True``.")
            return T
            
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='array',
                            items=[{'type': 'string'}]*3)


class GenericType(Type):
    class Data(SerializedData):
        baseT: Type
        args: List[Type]

        def encode(T: typing._GenericAlias) -> Union[GenericType.Data,Type.Data]:
            if T.__parameters__:  # __parameters__ is the list of non specified types
                if not all(isinstance(argT, typing.TypeVar) for argT in T.__args__):
                    raise NotImplementedError(
                        "We only support generic types for which either non of the "
                        "type arguments are specified, or all of them are.\n"
                        f"Type {T} has both.")
                # For non-concrete types, the standard Type serialization format suffices
                # (NB: We can't use Type.reduce, because we need '_name' instead of '__name__'
                # FIXME: I don't think returning a Data of another type is supported yet
                return Type.Data(module=T.__module__, name=T._name)
                # raise NotImplementedError("Only concrete generic types can be serialized. "
                #                           "(So e.g. `List[int]`, but not `List`.)")
            if T.__module__ == "__main__":
                raise ValueError("Can't serialize types defined in the '__main__' module.")
            return T, T.__args__

        def decode(data: GenericType.Data) -> GenericType:
            return data.baseT[tuple(data.args)]

class PydanticGenericType(GenericType):
    class Data(GenericType.Data):
        def encode(T: pydantic.main.ModelMetaclass) -> PydanticGenericType.Data:
            # Get the base Generic type and type arguments
            #   E.g. For `Foo[int]`, retrieve `Foo` and `(int,)`
            # NB: In contrast to normal generic types, Pydantic Generics don't have
            #   __origin__ or __args__ attributes. But Pydantic maintains a cache
            #   of instantiated generics, to avoid re-instantiating them; this
            #   cache is keyed by the base generic type, and the argument types
            try:
                genT, paramT = next(k for k, v in pydantic.generics._generic_types_cache.items()
                                    if v is T)
            except StopIteration:
                # The cache is updated as soon as a concrete generic type is first created,
                # i.e. the first time `Foo[int]` appears.
                # The only way it should happen that T is not in the cache, is if
                # T is a pure generic type. In this case, the normal Type serializer
                # works fine.
                return Type.Data.encode(T)
            
            return (genT, paramT)
