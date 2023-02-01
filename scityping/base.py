import abc
import logging
from typing import Union, Type, Any, List, Tuple
from collections.abc import Callable as Callable_, Sequence as Sequence_
from dataclasses import asdict, is_dataclass
import inspect
from .utils import get_type_key, TypeRegistry

logger = logging.getLogger(__name__)

__all__ = ["json_like", "Serializable", "Serialized"]

# ##############
# Custom JSON objects
# We use the following convention for JSON serializations: objects are
# serialized as tuples, where the first entry is a unique string identifying
# type. Thus we can check this string to know whether we should attempt decoding.

def json_like(value: Any, type_str: Union[str,List[str]],
              case_sensitive: bool=False):
    """
    Convenience fonction for checking whether a serialized value might be a
    object serialized with `scityping.base.json_encoder`. 
    :param:value: The value for which we want to determine if it is a
        JSON-serialized object.
    :param:type_str: The type string of the type we are attempting to
        deserialize into. May be in list, in which the function returns true
        if any one matches.
    :param:case_sensitive: Whether the comparison to `type_str` should be
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
# Base class for custom serializable types

# NB: In pydantic.py, we associate the `Serializable.json_encoder` to the *abstract* base class `ABCSerializable`.
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
    # (and so provide `Data`, `json_encoder`, `__get_validators__`, etc.)
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
            # Check that 'encode' is a classmethod (https://stackoverflow.com/a/19228282)
            if (not hasattr(cls.Data, "encode")
                  or not isinstance(cls.Data.encode, Callable_) 
                  # or not inspect.ismethod(cls.Data.encode)
                  # or cls.Data.encode.__self__ is not cls.Data  # NB: This does not prevent subclassing Data to reuse the encode
                  ):
                raise TypeError(f"{cls} is a subclass of `Serializable`, and "
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
        basecls = next(C for C in type.mro(cls)[::-1]
                       if C.__qualname__.lower() == name)
        if basecls is not cls:
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
            if issubclass(C, Serializable):
                C._registry[T] = cls


    @classmethod  # Pydantic validation hook
    def __get_validators__(cls):
        yield cls.validate

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
                    json = subcls.json_encoder(value)
                    subcls.validate(json)
                return subcls.validate(subcls.json_encoder(value))  # NB: json_encoder() doesn’t do the actual cast to string, so this isn’t too expensive

        # Branch 3: Check if `value` is an instantiated `Data` object.
        #           If so, construct the target type from it.
        types_with_matching_dataclasses = [
            subcls for subcls in set(cls._registry.values())  # We may have duplicate entries (e.g. point both 'Generator' and 'NPGenerator' to NPGenerator)
            if isinstance(subcls.Data, type) and isinstance(value, subcls.Data)]   # The first check is mostly for Data given as an abstractmethod/property in an abstract base class
        if types_with_matching_dataclasses:
            assert len(types_with_matching_dataclasses) == 1, "There should never be more than one matching Data class."
            targetT = types_with_matching_dataclasses[0]
            decoder = getattr(targetT.Data, 'decode', None)
            if decoder:
                return decoder(value)
                # Here we skip the call to `validate`: If targetT defines a `decode`,
                # then it expects its value to be used as-is.
                # This allows `decode` to be used to define a different type
                # (e.g. `range` instead of `Range`) without causing an infinite
                # loop (since then the code does not exit on branch #1).
            else:
                kwds = asdict(value) if is_dataclass(value) else value.dict()
                    # Support both dataclass and Pydantic BaseModel for the nested Data class
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

            targetT = cls._registry[value[0]]
            if isinstance(value[1], targetT.Data):  # This can happen if we call `json_encoder` but never actually convert the result to string
                data = value[1]
            elif isinstance(value[1], tuple):
                data = targetT.Data(*value[1])
            elif isinstance(value[1], dict):
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
            # try:
            #     return cls(value)
            # except Exception as e:
            # Give up and print an error message
            valuestr = str(value)
            if len(valuestr) > 1000: valuestr = valuestr[:999] + "…"
            raise TypeError(f"Unable to validate the received value to the type {cls}."
                            f"    Type received: {type(value).__qualname__}; registry key: '{type(value).__qualname__.lower()}'\n"
                            f"    Value received: {valuestr}\n\n")
                            # f"Attempting `{cls}(<value>)` raised the "
                            # f"following error:\n{e}")
    @classmethod
    def json_encoder(cls, value, *args, **kwargs) -> Tuple[str, "Data"]:
        """
        Generic custom serializer. All serialized objects follow the format
        ``(type name, serialized data)``. For example,
        ``(scityping.range, {'start': 0, 'stop': 5, 'step': none})``.
        The type name is inferred from the `value`’s type and is case insensitive.
        It is the key generated by `scityping.utils.TypeRegistry`.

        The class used to serialize `value` is inferred by matching the type
        of `value` to one of the entries in `cls._registry`.

        Additional positional and keyword arguments are passed on to
        ``S.Data.encode(*args, **kwargs)``, where ``S`` is the class used to
        serialize `value`. It is the user's responsibility to ensure that these
        are valid for the inferred encoder.
        """
        # This first line deals with the case when `json_encoder` is called with
        # the parent class (typically `Serializable`), or when the serializer
        # is attached to a base class (e.g. the same serializer for all subclasses of np.dtype) 
        for C in type(value).mro():
            serializer_cls = cls._registry.get(C)
            if serializer_cls:
                break
        else:
            # I'm not sure what the most appropriate error type should be, but at least TypeError is consistent with the similar error in `validate()`
            # Also `KeyError` is special and doesn't display newlines; I think it is best reserved for cases where the user interacts directly with a mapping
            raise TypeError("There seems to be no JSON encoder registered for any "
                           f"of the types {type(value).mro()}. Note that types "
                           "depending on external librairies (like scipy or torch) "
                           "are only registered when the corresponding module in "
                           "scityping is imported.\n"
                           f"Attempted to serialize using: {cls}.json_encoder\n"
                           f"Registered types for this class: {cls._registry.keys()}")

        data = serializer_cls.Data.encode(value, *args, **kwargs)
        if not isinstance(data, serializer_cls.Data):
            # To reduce boilerplate, we allow `encode` to return a tuple or dict
            # These are respectively treated as positional or keyword args to Data
            if isinstance(data, tuple):
                try:
                    data = serializer_cls.Data(*data)
                except TypeError as e:
                    if e.args[0].startswith("__init__() takes exactly 1 positional argument"):
                        # We can end up here if we try to initialize a Pydantic BaseModel with positional arguments.
                        # As a convenience, if Data class has a `__fields__` attribute, we will use that to construct a dictionary
                        fields = getattr(serializer_cls.Data, "__fields__", None)
                        if fields is None:
                            raise e  # We are not in a BaseModel - re-raise the original error
                        kwds = {k: v for k, v in zip(fields, data)}
                        data = serializer_cls.Data(**kwds)
                    else:
                        raise e

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


class SerializedGen():
    """
    Convenience constructor for a type describing serialized objects.
    If ``MyType`` is a subclass of `Serializable`, then ``Serialized[MyType]``
    returns the type ``Tuple[str, T.Data]``.
    """
    # Make a singleton
    __instance = None
    def __new__(cls):
        return SerializedGen.__instance or super().__new__(cls)
    # Actual implementation
    def __getitem__(self, T: Type[Serializable]) -> Type:
        if not isinstance(T, type):
            raise TypeError("Serializable[…] only accepts types which are "
                            f"subclasses of Serializable. {T} is not even a type.")
        elif not issubclass(T, Serializable):
            raise TypeError("Serializable[…] only accepts types which are "
                            "subclasses of Serializable. "
                            f"{T} does not subclass Serializable.")
        return Tuple[str, T.Data]

Serialized = SerializedGen()