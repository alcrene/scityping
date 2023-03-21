from __future__ import annotations

import abc
import logging
import textwrap
import inspect
from typing import ClassVar, Union, Type, Any, Literal, Callable, List, Tuple
from collections.abc import (
    Callable as Callable_, Sequence as Sequence_, Iterable as Iterable_,
    Mapping as Mapping_)
from dataclasses import fields, is_dataclass
from types import FunctionType
from .utils import get_type_key, TypeRegistry
from .typing import StrictStr, StrictBytes, StrictInt, StrictFloat, StrictBool

try:
    from pydantic import BaseModel
except ModuleNotFoundError:
    # BaseModel is used only for isinstance checks – if pydantic is not loaded,
    # those tests are False by default. Therefore instantiating a dummy class suffices
    class BaseModel:
        def __new__(cls):
            raise RuntimeError("This dummy BaseModel is not meant to be instantiated.")

logger = logging.getLogger(__name__)

class MISSING:  # Sentinel value when looking for a value in the namespace
    pass

__all__ = ["json_like", "Serializable", "Serialized", "Dataclass"]

# ##############
# Custom JSON objects
# We use the following convention for JSON serializations: objects are
# serialized as tuples, where the first entry is a unique string identifying
# type. Thus we can check this string to know whether we should attempt decoding.

# TODO: Rename to `reduce_like` (or something along those lines)
def json_like(value: Any, type_str: Union[str,List[str]],
              case_sensitive: bool=False):
    """
    Convenience fonction for checking whether a serialized value might be a
    object serialized with `scityping.base.reduce`.

    Args:
        value: The value for which we want to determine if it is a
            JSON-serialized object.
        type_str: The type string of the type we are attempting to
            deserialize into. May be in list, in which the function returns true
            if any one matches.
        case_sensitive: Whether the comparison to `type_str` should be
            case-sensitive.
    """
    if isinstance(type_str, str):
        type_str = [type_str]
    casefold = (lambda v: v) if case_sensitive else str.casefold
    return any(
        (not isinstance(value, str) and isinstance(value, Sequence_) and value
         and isinstance(value[0], str) and casefold(value[0]) == casefold(type_str_))
        for type_str_ in type_str)

# ###############
# Recursive reduction

def deep_reduce(v, **kwargs):
    from .functions import serialize_function  # HACK !! We should put this in utils or something, or better – avoid the need to special case Callables at all
    return (v.deep_reduce(v, **kwargs) if isinstance(v, Serializable)
            else Dataclass.deep_reduce(v, **kwargs) if (is_dataclass(v) and not isinstance(v, type))
            else serialize_function(v) if isinstance(v, FunctionType)  # Only serialize plain functions – not callable classes, methods, lambdas, etc.
            else v)

# ###############
# Base class for custom serializable types

# NB: In pydantic.py, we associate the `Serializable.reduce` to the *abstract* base class `ABCSerializable`.
class ABCSerializable(abc.ABC):
    """
    This is the class added to the list of JSON encoders.
    Any type can be registered as a virtual subclass, which will then divert
    its serialization to the hierarchy managed by `Serializable`.

    This abstract class is used by `Serializable` to register parent types
    of the same name, so that they are also serializable.

    (The reason we need a separate class, is that only abstract classes allow
    virtual subclasses, but we don't want to make `Serializable` abstract.)
    """
    # Registry of types which are registered as virtual subclasses of ABCSerializable.
    # This mirrors but differs from Serializable._registry: Both have the same
    # keys, but the values in the latter are true subclasses of Serializable
    # (and so provide `Data`, `reduce`, `__get_validators__`, etc.)
    # It should always be true that an entry in Serializable._registry is a
    # non-strict subclass of the corresponding entry in ABCSerializable._registry.
    # NB: Keys are the *types* (values) of Serializable._registry; this is to
    #     allow subclasses of `Serializable` to redefine _registry to limit the
    #     accepted sub classes.
    _base_types = {}

# NB: Conceptually this is an abstract class, but since it is intended for use
# as a mixin, we don't inherit abc.ABC (the validation functionality is reimplemented
# in `__init_subclass__` instead). Otherwise subclasses would need to implement
# custom metaclasses whenever their parents aren't also ABC subclasses.
@ABCSerializable.register  # Registering Serializable allows any subclass of Serializable to be recognized, without having to register each individually
class Serializable:
    """
    An abstract base class used to standardize the definition of
    Pydantic-serializable types, and reduce the boilerplate to do so.

    A `Serializable` subclass `SubT` must define a nested class `Data`, which
    inherits from `scityping.BaseModel` and defines the data structure used to
    serialize the class. The attribute names of the `Data` class must match the
    keywords of the class' ``__init__``.

    The nested `Data` class must also define a method `encode`, which
    takes an instance of `SubT`, extracts the attributes required for
    serializing it, and returns them as a `SubT.Data` instance.

    .. hint::
       If a `Serializable` has *the same name* (case insensitive) as one of
       its parent classes, then any instance of the *parent* class will be
       serializable. Specifically, the class' `__qualname__` attribute is
       used for this somewhat magical behaviour.

    .. admonition:: Pickle support

       Serializable objects automatically support pickling if their
       `Data.encode` method returns something that is pickleable.
       Note that this only applies to *subclasses* of `Serializable`; many of
       the provided types in this package (including `~scityping.numpy.Array`,
       `~scityping.scipy.Distribution`) *don't* return instances of themselves
       when validating values, instead returning clean, unmodified base types
       (the stated examples return `numpy.ndarray` and `scipy.stats.<rv_frozen>`
       respectively). For these types, pickling if and only if the base types
       supports it.

       Among the types defined in this package,
       `~scityping.functions.PureFunction` and its subclasses is probably the
       one which most benefits from pickling support.
    """
    # Subclasses are stored in a registry
    # This allows deserializers to be retrieved based on the serialized name.
    # Each subclass creates its own _registry, and adds itself to the registry
    # of all its parents
    _registry = TypeRegistry()

    # Subclass must define a nested class `Data`
    @property
    @classmethod
    @abc.abstractmethod
    def Data(self):
        # Which itself defines a class method `encode`
        # @classmethod
        # @abc.abstractmethod
        # def encode(cls, value) -> Data:
        #     raise NotImplementedError
        raise NotImplementedError

    def __init_subclass__(cls):
        ## Reproduce ABC functionality, since we don't subclass ABC ##
        if not inspect.isabstract(cls):
            if getattr(cls.Data, '__isabstractmethod__', False):
                raise TypeError(f"Can't instantiate class {cls.__qualname__} with "
                                "undefined `Data`.")
            # Ensure that all subclasses define their own `Data` class
            # (Simply inheriting Data from the parent class makes deserialization ambiguous.)
            if (any(cls.Data is getattr(C, "Data", None) for C in type.mro(cls)[1:])
                and not issubclass(cls, Dataclass)):  # Dataclass has special support to allow it to be used as the type for Data without causing infinite recursion
                raise TypeError(f"The `Serializable` type {cls.__qualname__} must "
                                "define its own `Data` class.")
            # Check that 'encode' is a classmethod (https://stackoverflow.com/a/19228282)
            if (not hasattr(cls.Data, "encode")
                  or not isinstance(cls.Data.encode, Callable_) 
                  # or not inspect.ismethod(cls.Data.encode)
                  # or cls.Data.encode.__self__ is not cls.Data  # NB: This does not prevent subclassing Data to reuse the encode
                  ):
                raise TypeError(f"{cls.__qualname__} is a subclass of `Serializable`, and "
                                "therefore its nested `Data` class must define "
                                "a class method `encode`.")
        ## Update subclass registries ##
        # Create a new registry for this class, and add to all the parents
        cls._registry = TypeRegistry()
        for C in type.mro(cls):  # In contrast to `cls.mro()`, works also with metaclasses
            if issubclass(C, Serializable):
                C._registry[cls] = cls
        # Update the registry of base clasess
        # We keep registries of both the base classes (for isinstance checks)
        # and the serializable class (for serialization functions).
        # See /scityping/docs/implementation_logic
        # (Conceptually, one can think of _base_types as a registry of super classes,
        # and _registry as a registry of subclases, perhaps we could get away with iterating through _registry ?)
        ABCSerializable._base_types[cls] = (cls,)
        # MAGIC: Register the lowest class in the MRO OF THE SAME NAME as a virtual subclass.
        #        This is to allow adding serializers for existing data types.
        name = cls.__qualname__.lower()
        try:
            basecls = next(C for C in type.mro(cls)[::-1]
                           if C.__qualname__.lower() == name
                              and not issubclass(C, Serializable))
            # Consider the case here where we define a plain Dataset in `base`, then
            # define `C = Dataset(base.Dataset, torch...Dataset). We want to
            # register the `torch...Dataset` with C. (Any parent which
            # (is Serializable will already have been registered above.)
        except StopIteration:
            pass
        else:
            cls.register(basecls)

    @classmethod
    def register(cls, T:type):
        """Register type `T` as serializable by `cls`."""
        if not issubclass(T, Serializable):
            # Registering everything serializable as virtual subclass would do no harm, but seems unclean (and might cause a performance hit)
            ABCSerializable.register(T)
        # Update _base_types with a minimal set of base types, including `T`
        ABCSerializable._base_types[cls] = tuple(
            base for base in ABCSerializable._base_types[cls]
            if not issubclass(base, T)) + (T,)
        # Tell all the parents of `cls` that they can serialize `T`.
        for C in type.mro(cls):
            if issubclass(C, Serializable) and T not in C._registry:
                C._registry[T] = cls


    @classmethod  # Pydantic validation hook
    def __get_validators__(cls):
        yield cls.validate

    def __reduce__(self):  # Pickle serialization hook
        return (self.validate, (self.deep_reduce(self),))

    @classmethod
    def validate(cls, value, field=None):  # `field` not currently used: only there for consistency
        """
        Validate `value` to an instance of `cls` (possibly of a subclass).
        If it is already of this type, return `value` unchanged.
        Otherwise apply type coercion or deserialization, as required.
        If nothing succeeds, raise `TypeError`.
        For precise details, see /scityping/docs/implementation_logic#validation-logic
        """
        # Branch 1: Check if `value` is already of a serializable type.
        #           If so, simply return `value` unchanged.
        if any(isinstance(value, subcls)
               for subcls in cls._registry.values()):
            return value

        # Branch 2: Check if it matches one of the base types wrapped by serializers
        #           If so, cast it to the corresponding subclass by calling `subclass.validate` on `value`.
        base_types = ABCSerializable._base_types
        matching_basesubclasses = {subcls: set(type.mro(subcls))
                                   for subcls in cls._registry.values()
                                   if isinstance(value, base_types[subcls])}
        if matching_basesubclasses:
            # Remove any subcls which is a strict superclass of an also matching subclass
            to_remove = []
            for subcls, mro in matching_basesubclasses.items():
                if any(mro < othermro for othermro in matching_basesubclasses.values()):
                    to_remove.append(subcls)
            for subcls in to_remove:
                del matching_basesubclasses[subcls]
            assert len(matching_basesubclasses) > 0, "Bug in the logic removing less specific Serializable types."
            if len(matching_basesubclasses) > 1:
                raise NotImplementedError(
                    "Diamond inheritance between Serializable subclasses is not yet "
                    f"supported.\nType {type(value)} subclasses {matching_basesubclasses}.")
            else:
                # Recast value to be of type 'cls'
                subcls = next(iter(matching_basesubclasses))
                if "generator" in str(value).lower():
                    json = subcls.reduce(value)
                    subcls.validate(json)
                return subcls.validate(subcls.reduce(value))  # NB: reduce() doesn’t do the actual conversion to string, so this isn’t too expensive

        # Branch 3: Check if `value` is an instantiated `Data` object.
        #           If so, construct the target type from it.
        types_with_matching_dataclasses = [
            subcls for subcls in set(cls._registry.values())  # We may have duplicate entries (e.g. point both 'Generator' and 'NPGenerator' to NPGenerator)
            if isinstance(subcls.Data, type) and isinstance(value, subcls.Data)]   # The first check is mostly for Data given as an abstractmethod/property in an abstract base class
        if types_with_matching_dataclasses:
            #   `types_with_matching_dataclasses` may have multiple entries when we specialize a type.
            #   For example, if we specialize `Website` into `CommerceWebsite`
            #       class Website(Serializable):
            #         class Data:
            #           ...
            #       class CommerceWebsite(Website):
            #         class Data(Website.Data)
            #           ...
            #   both `Website` and `CommerceWebsite` will be in `BaseWebsite._cls_registry`.
            #   Two different situations can cause multiple matches
            #   1) `CommerceWebsite` does not define a nested `Data` at all and we call
            #          Website.validate(website_data: Website.Data)
            #      This is prohibited by __init_subclass__ because it cannot be resolved.
            #   2) `CommerceWebsite` subclasses `Website.Data`, as in the example above, and we call
            #          Website.validate(website_data: CommerceWebsite.Data)
            #      In this case we want to deserialize to the most specific type: CommerceWebsite
            #   Only when there is multiple inheritance between Serializable types should we need to raise an error.

            # # Remove matching types which have strict subclasses
            # for C in types_with_matching_dataclasses[:]:
            #     if any(issubclass(D, C)
            #            for D in types_with_matching_dataclasses if D is not C):
            #         types_with_matching_dataclasses.remove(C)
            assert len(types_with_matching_dataclasses) == 1, \
                f"Unable to resolve the target data type for value of type {type(value)}; " \
                f"all these Serializable types match: {types_with_matching_dataclasses}." \
                "This can happen if the types’ Data classes have diamond inheritance"
            targetT = types_with_matching_dataclasses[0]
            decoder = getattr(targetT.Data, 'decode', None)
            # If `value` is a plain dataclass, apply our basic support for deserialization
            if (is_dataclass(value) and not isinstance(value, type)
                  and not hasattr(value, "__pydantic_model__")):
                value = validate_dataclass(value, inplace=True)
            # Use either custom or default decoder
            if decoder:
                return decoder(value)
                # Here we skip the call to `validate`: If targetT defines a `decode`,
                # then it expects its value to be used as-is.
                # This allows `decode` to be used to define a different type
                # (e.g. `range` instead of `Range`) without causing an infinite
                # loop (since then the code does not exit on branch #1).
            else:
                # We support dataclass, Pydantic BaseModel and mappings for the nested Data class
                if is_dataclass(value) and not isinstance(value, type):
                    # Make a shallow copy of the fields (`dataclasses.asdict` copies everything, and recurses into dataclasses, dicts, lists and tuples)
                    kwds = {field.name: getattr(value, field.name) for field in fields(value)}
                elif hasattr(value, "__fields__"):
                    # Assume a Pydantic BaseModel
                    kwds = dict(value)
                elif hasattr(value, "items"):
                    kwds = dict(value.items())
                else:
                    raise TypeError(f"Nested `Data` type for {targetT.__qualname__} "
                                    "should be either a dataclass, a Pydantic BaseModel "
                                    f"or a mapping. inheritance for {targetT.__qualname__}:\n"
                                    f"{targetT.Data.mro()}.")
                obj = targetT(**kwds)
                # If targetT has an overridden `validate`, ensure it is executed on the serialized data
                # If `validate` is not overridden, this will exit via branch #1
                return targetT.validate(obj)

        # Branch 4: Check if `value` is a serialized `Data` object.
        #           If so, deserialize it and call `targetT.validate`
        #           (which will use Branch 3 above, unless `validate` is overridden).
        elif (isinstance(value, Sequence_)
              and len(value) == 2
              and isinstance(value[0], str)):
            targetT = cls._registry.get(value[0])
            if targetT is None:
                valuestr = str(value)
                if len(valuestr) > 1000: valuestr = valuestr[:999] + "…"
                raise TypeError("Serialized data does not match any of the registered "
                                f"`Serializable` subclasses of {cls}.\n"
                                f"    Serializable subclasses: {list(cls._registry)}\n"
                                f"    Type received: {type(value).__qualname__}; registry key: '{type(value).__qualname__.lower()}'\n"
                                f"    Value received: {value}\n\n")

            if isinstance(value[1], targetT.Data):  # This can happen if we call `reduce` but never actually convert the result to string
                data = value[1]
            elif isinstance(value[1], Sequence_):
                data = targetT.Data(*value[1])
            elif isinstance(value[1], Mapping_):
                data = targetT.Data(**value[1])
            else:
                raise TypeError(f"Serialized {cls.__qualname__}.Data must be either "
                                "a tuple (positional args) or dict (keyword args). "
                                f"Received {type(value[1])}.")
            return targetT.validate(data)  # Unless overriden is targetT, this will use the `matching_dataclasses` branch above

        # Branch 5: As a final option, try simply passing the value as argument to this type
        #           This always on the base Serializer class, since it takes no arguments,
        #           but it allows things like ``PintQuantity.validate(3.)``
        # EDIT: Don't try passing as args, because this can prevent exceptions
        #   from being raised (e.g. an array will accept a heterogeneous list).
        #   Pydantic in particular relies on the "wrong" types raising exceptions,
        #   in order to try the next type in a list.
        else:
            # Give up and print an error message
            valuestr = str(value)
            if len(valuestr) > 1000: valuestr = valuestr[:999] + "…"
            raise TypeError(f"Unable to validate the received value to the type {cls}."
                            f"    Type received: {type(value).__qualname__}; registry key: '{type(value).__qualname__.lower()}'\n"
                            f"    Value received: {valuestr}\n\n")
                            # f"Attempting `{cls}(<value>)` raised the "
                            # f"following error:\n{e}")
    @classmethod
    def reduce(cls, value, **kwargs) -> Tuple[str, "Data"]:
        """
        Generic custom serializer. All serialized objects follow the format
        ``(type name, serialized data)``. For example,
        ``(scityping.range, {'start': 0, 'stop': 5, 'step': none})``.
        The type name is inferred from the `value`’s type with `get_type_key`
        and is case insensitive. It is the key used by `scityping.utils.TypeRegistry`.

        The class used to serialize `value` is inferred by matching the type
        of `value` to one of the entries in `cls._registry`.

        Additional positional and keyword arguments are passed on to
        ``S.Data.encode(*args, **kwargs)``, where ``S`` is the class used to
        serialize `value`. Only `kwargs` which match an argument in the signature
        of `S.Data.encode` are passed on; this allows `kwargs` to contain encoding
        arguments for different encoders, as long as all encoders use different
        argument names. (The use case for this filtering is `deep_reduce`, which
        through recursion may encode many objects but `kwargs` can only be
        specified once.)
        """
        # This first line deals with the case when `reduce` is called with
        # the parent class (typically `Serializable`), or when the serializer
        # is attached to a base class (e.g. the same serializer for all subclasses of np.dtype) 
        for C in type.mro(type(value)):  # In contrast to `type(value).mro()`, this works also when `value` is a type
            serializer_cls = cls._registry.get(C)
            if serializer_cls:
                break
        else:
            # I'm not sure what the most appropriate error type should be, but at least TypeError is consistent with the similar error in `validate()`
            # Also `KeyError` is special and doesn't display newlines; I think it is best reserved for cases where the user interacts directly with a mapping
            raise TypeError("There seems to be no encoder registered for any "
                           f"of the types {type.mro(type(value))}. Note that types "
                           "depending on external librairies (like scipy or torch) "
                           "are only registered when the corresponding module in "
                           "scityping is imported.\n"
                           f"Attempted to serialize using: {cls}.reduce\n"
                           f"Registered types for this class: {cls._registry.keys()}")

        # Call Data.encode, passing any parameters from kwargs that match a parameter in Data.encode.
        if kwargs:
            encode_kwargs = inspect.signature(serializer_cls.Data.encode).parameters.keys()
            kwargs = {kw: arg for kw, arg in kwargs.items() if kw in encode_kwargs}
        data = serializer_cls.Data.encode(value, **kwargs)

        # NB: To reduce boilerplate, we allow `encode` to return a tuple or dict
        # These are respectively treated as positional or keyword args to Data
        if not isinstance(data, serializer_cls.Data):
            if isinstance(data, tuple):
                if issubclass(serializer_cls.Data, BaseModel):
                    # As a convenience, if Data class is a BaseModel, it has a
                    # `__fields__` attribute we can use to construct an argument dictionary
                    fields = serializer_cls.Data.__fields__
                    kwds = {k: v for k, v in zip(fields, data)}
                    data = serializer_cls.Data(**kwds)
                else:
                    data = serializer_cls.Data(*data)

            elif isinstance(data, dict):
                data = serializer_cls.Data(**data)
            else:
                data = serializer_cls.Data(data)
                # raise TypeError(
                #     f"{serializer_cls.__qualname__}.Data.encode method returned an object of "
                #     f"type '{type(data)}'. It must return either an instance of "
                #     f"{serializer_cls.__qualname__}.Data, a tuple or a dict. If a tuple or "
                #     f"dict, these must be arguments to {serializer_cls.__qualname__}.Data.")
        return (get_type_key(serializer_cls), data)

    @classmethod
    def deep_reduce(cls, value, **kwargs) -> Tuple[str, Union[dict,tuple]]:
        """
        Call `reduce` recursively, converting each result to either a tuple
        or a dictionary. The returned value therefore only contains types
        which can be serialized without `scityping`.
        If certain encoders (i.e. the ``Data.encode`` methods) take additional
        parameters, these can be passed as keyword arguments. To avoid
        unexpected values, different encoders should take care to use different
        argument names.
        """
        type_key, data = cls.reduce(value, **kwargs)

        if is_dataclass(data):
            reduced_type = dict
            data_iter = ((k,getattr(data,k)) for k in data.__dataclass_fields__)
        elif isinstance(data, BaseModel):
            reduced_type = dict
            data_iter = data
        elif isinstance(data, Mapping):
            reduced_type = dict
            data_iter = data.items()
        elif isinstance(data, Iterable_):
            reduced_type = tuple
            data_iter = data
        else:
            raise TypeError(f"`{cls.__qualname__}.reduce` returned an object of unexpected "
                            f"type '{type(data).__qualname__}', which is incompatible with "
                            f"`{cls.__qualname__}.deep_reduce`.")

        if reduced_type is dict:
            reduced_data = {k: deep_reduce(v)
                            for k, v in data_iter}
        else:
            reduced_data = tuple(deep_reduce(v) for v in data_iter)

        return (type_key, reduced_data)


class SerializedMeta(type):
    # # Make a singleton
    # __instance = None
    # def __new__(cls):
    #     return SerializedGen.__instance or super().__new__(cls)
    # Actual implementation
    def __getitem__(cls, T: Type[Serializable]) -> Type:
        if cls is not Serialized:
            raise RuntimeError("Type definitions of the form `Serialized[…][…]` "
                               "are not supported.")
        elif not isinstance(T, type):
            raise TypeError("Serialized[…] only accepts types which are "
                            f"subclasses of Serializable. {T} is not even a type.")
        elif not issubclass(T, Serializable):
            raise TypeError("Serialized[…] only accepts types which are "
                            "subclasses of Serializable. "
                            f"{T} does not subclass Serializable.")
        # return Tuple[str, T.Data]
        return SerializedMeta(f"Serialized{T.__name__}", (Serialized,),
                              {"_serialized_type": T, "__slots__": ()})

class Serialized(tuple, metaclass=SerializedMeta):
    """
    Constructor for a type describing serialized objects.
    If ``MyType`` is a subclass of `Serializable`, then ``Serialized[MyType]``
    returns a type which is essentially ``Tuple[str, T.Data]`` but which also:
    - accepts serialized data from subclasses of ``MyType``;
    - accepts instances of ``MyType`` (in which case they are serialized).
    """
    __slots__ = ()
    _serialized_type: ClassVar

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, value):
        # If we got `MyType` instead of `Serialized[MyType]`, convert it to the right form.
        if isinstance(value, cls._serialized_type):
            value = cls.reduce(value)
        # Raise error if value is not of correct type
        if ( not isinstance(value, Sequence_) or len(value) != 2 or not isinstance(value[0], str) ):
            value_str = str(value)
            if len(value_str) > 150: value_str = value_str[:149] + "…"
            raise TypeError(f"A value matching Serialized[{cls._serialized_type.__qualname__}] "
                            "should be a tuple of length two, where the first element is a string. "
                            f"Received:\n {value_str}")
        # Determine the target type for the data by inspecting the first arg.
        # This might not be the same as `_serialized_type`: it could a subclass
        targetT = cls._serialized_type._registry.get(value[0])
        if targetT is None:
            raise TypeError(f"'{value[0]}' does not match any of the subtypes "
                            f"registered to {cls.__qualname__}.")
        # targetT.Data might be a BaseModel, dataclass, TypedDict, etc.
        # Also value[1] might be serialized or not.
        # So we inspect the type and value to determine how to return the tuple (key, Data)
        if isinstance(value[1], targetT.Data):
            # Nothing to do
            return cls(value)
        elif hasattr(targetT.Data, "validate"):
            return cls((value[0], targetT.Data.validate(value[1])))
        elif hasattr(targetT.Data, "__get_validators__"):
            data = value[1]
            for validator in targetT.Data.__get_validators__():
                data = validator(data)
            return cls((value[0], data))
        elif isinstance(value[1], dict):
            return cls((value[0], targetT.Data(**value[1])))
        elif isinstance(value[1], Iterable_):
            return cls((value[0], targetT.Data(*value[1])))

#####################
# Dataclass
#####################
# (Included here instead of base_types to allow import by scityping.json.py)

import builtins
import numbers
import re
from inspect import getmodule
from dataclasses import dataclass, asdict, FrozenInstanceError
from typing import Union, Dict, List

# The `Dataclass` type is special cased: it is identified with `is_dataclass`
# rather than with `isinstance`, so we don't need to register a base class.
# (Nor could we, since there is no base dataclass)

dc_val_msg = ("Note that `scityping` provides only basic deserialization "
              "capabilities for plain dataclasses defined in the standard "
              "library. For richer types or to customize the validation, "
              "please use e.g. a Pydantic BaseModel or Pydantic dataclass.")

def split_type_str(s):
    """
    Split on ',', leaving inside of brakets untouched
    "List[int], Dict[str, float]" -> ["List[int]", "Dict[str, float]"]
    """
    els = []
    el_c = []
    counter = 0
    for c in s:
        if c == "[":
            counter += 1
        elif c == "]":
            if counter <= 0:
                raise RuntimeError(f"Unequal brackets in {s}")
            counter -= 1
        if c == "," and counter == 0:
            els.append("".join(el_c))
            el_c = []
        elif c == " ":
            pass  # Drop white space
        else:
            el_c.append(c)
    if counter != 0:
        raise RuntimeError(f"Unequal brackets in {s}")
    els.append("".join(el_c))
    return els

def get_type_annotation(T, ns: dict):
    if isinstance(T, str):
        # T may have an argument of the form Union[int,float]
        # The regex stores `Union` in m[1] and `[int,float]` in m[2]
        # It also succeeds on the following: `scityping.Array[np.uint32, 2]`
        m = re.match(r"([\w\.]+)(\[.*\])?", T)
        assert m is not None, f"Regex failed to parse the type '{T}'"
        T = ns.get(m[1], MISSING)
        if T is MISSING:
            # Some types allow literals in their argument (like Literal, or scityping.numpy.Array)
            # => Evaluate, then return as is. If this isn’t inside a type argument, an error will be raised later

            # Check for int or float: only numbers, possibly one '.'
            mT = re.match(r"\d+(\.\d*)?$", m.string)
            if mT: # int or float
                if mT[1] is None:  # No '.' => Int
                    T = int(m.string)
                else:
                    T = float(m.string)

            # Ellipsis
            elif m.string == "...":
                T = Ellipsis

            # Deref dotted names. If nothing is found, return the whole thing as a string
            else:
                els = m.string.split(".")
                obj = ns.get(els[0], MISSING)
                if obj is MISSING:
                    T = m.string
                else:
                    for el in els[1:]:
                        obj = getattr(obj, el, MISSING)
                        if obj is MISSING:  # Dotted name not found => return original type string 
                            T = m.string
                            break
                    else:  # All dotted names were found
                        T = obj
        elif m[2]:
            # Remove the start and end brackets
            arg = m[2][1:-1]
            # m = re.match(r"(\w+(?:\[[^\]]*\])?,? *)+", arg)
            # args = tuple(get_type_annotation(arg_str.strip(), ns)
            #              for arg_str in arg.split(","))
            args = tuple(get_type_annotation(arg_str.strip(), ns)
                         for arg_str in split_type_str(arg))
            if len(args) == 1:
                args = args[0]  # Required for types like Optional
            T = T[args]
    return T

def validate_dataclass_field(val, T: type):
    """
    .. rubric:: Supported types

    Scalar types

    - All subclasses of Serializable
    - int
    - float
    - str
    - bytes
    - bool
    - type (via base_types.Type)
    - tuple
    - dict
    - list
    - set
    - frozenset
    - scityping.NoneType

    Generic types

    - Union
    - Optional
    - Any
    - Tuple
    - List
    - Set
    - FrozenSet
    - Tuple
    - Literal (args must not be Serializable types)
    - Dict
    - Type (via base_types.Type)
    - Callable (accepts all callables; deserializes only PureFunction)

    """

    from . import base_types  # Imported here to avoid cyclic import

    __origin__ = getattr(T, "__origin__", None)

    ## Simplest case: `val` is already of type `T` ##
    # NB: This works even if `T` is not a deserializable type

    if __origin__ is None and isinstance(T, type) and isinstance(val, T):
        # (Something like Callable[[int],None] is still a type, but would
        #  raise TypeError in the isinstance (also we want to continue and 
        #  check the signature). The __origin__ skips generic types.
        return val

    ## Support for generic types ##

    # Union
    if __origin__ is Union:
        if isinstance(val, T.__args__):
            # Equivalent to smart_union: if value is already an instance of any type, don’t change it
            return val
        else:
            # Try coercing, going left to right in the types
            err_msgs = []
            for subT in T.__args__:
                try:
                    return validate_dataclass_field(val, subT)
                except TypeError as e:
                    err_msgs.append(str(e))
            else:
                # None of the types succeeded in deserializing: raise error, with all concatenated messages
                err_str = "\n".join(f"{subT} - {msg}"
                                    for subT, msg in zip(T.__args__, err_msgs))
                err_str = textwrap.indent(err_str, "  ")
                raise TypeError(f"Unable to validate to {T}. "
                                f"Error messages were:\n{err_str}")
    # List, Set, Frozenset
    elif __origin__ in (list, set, frozenset):
        assert len(T.__args__) == 1, \
            f"{'List' if __origin__ is list else 'Set' if __origin__ is set else 'FrozenSet'}[]" \
            " should have only one argument"
        return __origin__(validate_dataclass_field(el, T.__args__[0])
                          for el in val)
    # Tuple
    elif __origin__ is tuple:
        if T.__args__[-1] == ...:
            assert len(T.__args__) == 2, "Tuple[] should have only one type argument when it ends with '...'"
            return tuple(validate_dataclass_field(el, T.__args__[0])
                         for el in val)
        else:
            if len(val) != len(T.__args__):
                raise TypeError(f"Tuple field expects a value with length {len(T.__args__)}, "
                                f"but received one of length {len(val)}.")
            return tuple(validate_dataclass_field(el, elT)
                         for el, elT in zip(val, T.__args__))
    # Literal
    elif __origin__ is Literal:
        if val not in T.__args__:
            raise TypeError(f"Value {val} is not one of the values prescribed by {T}.")
        return val
    # Dict
    elif __origin__ is dict:
        assert len(T.__args__) == 2, "Dict[] should have exactly two arguments"
        keyT, valT = T.__args__
        return {validate_dataclass_field(k, keyT): validate_dataclass_field(v, valT)
                for k, v in val.items()}
    # Type
    elif __origin__ is base_types.Type or __origin__ is type:
        args = getattr(T, "__args__", None)
        if args:
            assert len(args) == 1, "Type[] should have only one argument."
            targetT = base_types.Type[args[0]]
            return targetT.validate(val)
        else:
            # A plain `typing.Type` with no argument will ĥave `__origin__` but no `__args__`
            return base_types.Type.validate(val)
    # Callable
    elif __origin__ is Callable_:
        from .functions import matching_signature, deserialize_function
        if isinstance(val, Callable_):
            # `val` is already a callable: just need to check signature
            if matching_signature(val, T):
                return val
            else:
                raise TypeError(f"Signature of function {val} does not match {T}.")
        elif (isinstance(val, Sequence_)
              and len(val) == 2
              and isinstance(val[0], str)):
            if "PureFunction" in val[0]:
                return Serializable.validate(val)
            else:
                raise TypeError(f"Serialized '{val[0]}' is not a recognized Callable type")
        elif isinstance(val, str):
            logger.warning("Calling `deserialize_function` without any  namespace: this "
                           "is likely to fail except for simple functions. For more reliable "
                           "deserialization, ensure functions are deserialized via `PureFunction`.")
            return deserialize_function(val)
        else:
            raise TypeError(f"{val} is not a Callable")

    ## Support for Any ##

    # Check if it looks like a Serializable type. If so, deserialize.
    # Otherwise, return as-is
    if T is Any:
        if (isinstance(val, Sequence_)
              and len(val) == 2
              and isinstance(val[0], str)):
            # TODO: Call directly the right branch, rather than repeat the checks in Serializable.validate
            return Serializable.validate(val)
        else:
            return val

    ## Support for builtin and Serializable types ##

    # From this point, we only support normal types (not those in typing.py)
    if not isinstance(T, type):
        raise TypeError(f"{T} does not support deserialization. {dc_val_msg}")

    # Since we know T is a type, we can use issubclass safely
    if issubclass(T, Serializable):
        return T.validate(val)
    # Subclasses of ABCSerializable don't have a `validate` method, so
    # we just treat them as a non-serializable type.
    # Could we do better ?
    elif T is StrictStr:
        if not isinstance(val, str): raise TypeError(f"{val} is not a string")
        return val
    elif T is StrictBytes:
        if isinstance(val, str): val = val.encode()  # Needed to allow deserialization from JSON
        elif not isinstance(val, bytes):  raise TypeError(f"{val} is not a bytes")
        return val
    elif T is StrictBool:
        if isinstance(val, str):
            if val.lower() == "true": val = True
            elif val.lower() == "false": val = False
            else: raise TypeError(f"{val} is not a bool")
        elif not isinstance(val, bool): raise TypeError(f"{val} is not a bool")
        return val
    elif T is StrictInt:
        if isinstance(val, str):
            if re.fullmatch(r"\d+", val): val = int(val)
            else: raise TypeError(f"{val} is not a integer")
        elif not isinstance(val, int): raise TypeError(f"{val} is not a integer")
        return val
    elif T is StrictFloat:
        if isinstance(val, str):
            if re.fullmatch(r"\d*\.\d*", val) and len(val) > 1: val = float(val)
            else: raise TypeError(f"{val} is not a float")
        elif not isinstance(val, float): raise TypeError(f"{val} is not a float")
        return val
    elif issubclass(T, numbers.Integral):
        return int(val)
    elif issubclass(T, numbers.Real):
        return float(val)
    elif issubclass(T, str):
        return str(val)
    elif issubclass(T, bytes):  # Emulate how Pydantic coerces bytes
        if isinstance(val, bytes):
            return val
        elif isinstance(val, str):
            return val.encode()
        elif isinstance(val, numbers.Number):
            return str(val).encode()
        if isinstance(val, bytearray):
            return bytes(val)
        else:
            raise TypeError(f"Coercion of values of type {type(val)} to "
                            "`bytes` is not supported.")
    elif issubclass(T, bool):
        return bool(val)
    elif issubclass(T, base_types.Type):
        T.validate(val)
    elif T is tuple:
        if isinstance(val, tuple):
            return val
        elif isinstance(val, Sequence_):
            return tuple(val)
        else:
            raise TypeError(f"Value {val} is not a tuple.")
    elif T in (list, set, frozenset):
        targetT = T
        if isinstance(val, targetT):
            return val
        elif isinstance(val, Sequence_):
            return TargetT(val)
        else:
            raise TypeError(f"Value {val} is not a {targetT.__name__}.")
    elif T is dict:
        if isinstance(val, dict):
            return val
        elif isinstance(val, Mapping_):
            return dict(val)
        else:
            raise TypeError(f"Value {val} is not a dict.")
    elif T is type(None):
        if val is None:
            return None
        else:
            raise TypeError("A field of type `NoneType` only accepts `None` values. "
                            f"Received {val}")
    else:
        # As a last attempt, check if the type provides a `validate` method.
        # This covers the types in base_types like `NoneType` and Number` which
        # don’t subclass Serializable
        validate = getattr(T, "validate", None)
        if validate:
            return validate(val)
        else:
            raise TypeError(f"Unrecognized type {T}. {dc_val_msg}")


def validate_dataclass(dc, inplace=False):
    """
    Args:
        dc: dataclass instance
        inplace: bool
            If `True`, attempt to modify the dataclass `dc` in place.
            If `False`, or `dc` is frozen, a new dataclass is created.

    .. Note:: Deserialization support for std lib dataclasses is purposefully
       limited to simple cases. Attempting to support every corner would
       quickly turn into a morass of conditionals, and duplicate a lot of the
       functionality of already mature libraries like Pydantic. Essentially
       only JSON types (str, int, float, bool) and `Serializable` types are
       supported.

       One reason we support dataclasses is because we use standard lib 
       dataclasses to serialize our own types is to avoid making Pydantic a
       hard dependency `scityping`. The features we need to do this are those
       which we support for validation.
    """
    # Retrieve the namespace in which to search for type definitions
    dc_mod = getmodule(dc)
    ns = {**builtins.__dict__, **(dc_mod.__dict__ if dc_mod else {})}
    new_kwds = {}
    # Validate each field
    for dc_field in fields(dc):
        _val = getattr(dc, dc_field.name)
        dc_field_type = dc_field.type
        # NB: field.type is often stored as a string
        if isinstance(dc_field_type, str):
            dc_field_type = get_type_annotation(dc_field_type, ns)

        _val = validate_dataclass_field(_val, dc_field_type)

        if inplace:
            try:
                setattr(dc, dc_field.name, _val)
            except FrozenInstanceError:
                inplace = False
                new_kwds[dc_field.name] = _val
        else:
            new_kwds[dc_field.name] = _val

    # Return the dataclass if it was modified in place, or create a new one
    if inplace:
        return dc
    else:
        return type(dc)(**new_kwds)

class Dataclass(Serializable):
    """
    This class provides very basic serialization/deserialization support for
    standard lib dataclasses.

    Supported types:

    - `int`, `float`, `str`, `bytes`
    - subclasses of `Serializable`

    NOT supported:

    - Union types, like `Union[int,float]`
    - List types, like `List[int]`
    - Pretty much anything not listed above.

    For more powerful deserialization capabilities, it is recommended to use
    full-featured serializable types, like Pydantic's `dataclass` and `BaseModel`.
    Nevertheless, this basic support may be useful

    - as a “first rung” of serialization dependencies: more complex Serializable
      types can use a plain `dataclass` as their `Data` class, as long as they
      limit themselves to the supported types;
    - to avoid adding a dependency on an external package like Pydantic;
    - when dealing with dataclass objects created by an external library, for
      which we cannot change the type.

    .. Note:: Because there is no base dataclass, dataclasses cannot be
       identified with the usual way of checking for a recognized base type.
       Their support is hard-coded and specific to dataclasses, using
       `is_dataclass` to recognized them. Since dataclasses may be used for
       the `Data` container, they also use custom code paths for deserialization
       to avoid recursion loops.

    .. Caution:: Use of plain dataclasses as field types for a serializable class
       is also limited. If `MyDataclass` is a plain dataclass type, then the
       following will deserialize as expected

       .. Code::Python
          class MyType(BaseModel):
            dc: MyDataclass

       but the following will not

       .. Code::Python
          class MyType(BaseModel):
            dcs: List[MyDataclass]

       Here again, if you need full serialization/deserialization support, you
       can use `scityping.pydantic.BaseModel` or `scityping.pydantic.dataclass`.
    """
    @classmethod
    def reduce(cls, dc, **kwargs):  # **kwargs required for cooperative signature
        return (get_type_key(cls), cls.Data.encode(dc))
    @classmethod
    def deep_reduce(cls, value, **kwargs):
        type_key, (T, dc_dict) = cls.reduce(value, **kwargs)
        reduced_dict = {k: deep_reduce(v, **kwargs) for k, v in dc_dict.items()}
        return (type_key, (deep_reduce(T, **kwargs), reduced_dict))
    # @classmethod
    # def __get_validators__(cls):
    #     yield cls.validate
    # We need to replace branch 3, since Data doesn't actually do deserialization
    @classmethod
    def validate(cls, value, field=None):
        # Branch 1: Subclasses which overwrite Data to set their own fields
        # (AFAIK, branch 2 should always suffice, but is more verbose since
        #  it wraps everything in an extra layer with 'type' and 'data')
        if (isinstance(value, cls.Data)
              and is_dataclass(cls)
              and cls.__dataclass_fields__.keys() <= value.__dataclass_fields__.keys()):
              # Additional tests needed for types which subclass Dataclass but don’t
              # redefine `Data`: in that case the fields are those of `Dataclass.Data`.
            value = validate_dataclass(value, inplace=True)
            dc_kwargs = {dc_field.name: getattr(value, dc_field.name)
                         for dc_field in fields(value)}
            return cls(**dc_kwargs)
        elif isinstance(value, Dataclass.Data):
            value = validate_dataclass(value, inplace=True)
            return value.type(**value.data)
        elif cls is Dataclass and is_dataclass(value):
            # The generic `Dataclass` serves as an ABC for all dataclasses
            return value
        else:
            # We should use another branch
            return super().validate(value, field)

    @dataclass
    class Data:
        type: Type
        data: Dict[str,Any]
        def encode(dc) -> Dataclass.Data:
            return (type(dc), asdict(dc))
