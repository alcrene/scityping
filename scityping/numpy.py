"""
Type annotations and serializers for NumPy types
"""

from __future__ import annotations

import abc
import logging
from types import SimpleNamespace
from typing import (Optional, Union, Any, Literal,
                    Iterable, Sequence, Tuple, List, Dict)
from dataclasses import asdict
from functools import lru_cache

from .base import json_like, Serializable, ABCSerializable, validate_dataclass
from .base_types import SerializedData
from .utils import LazyDict, get_type_key

try:
    from pydantic import StrictInt, StrictFloat
except ModuleNotFoundError:
    StrictInt, StrictFloat = int, float

import numpy as np

import io
# import blosc
import base64

cache = lru_cache(maxsize=None)  # Compatibility with ≤3.8
logger = logging.getLogger(__name__)

# ##################
# Helpers for NumPy types

def get_dtype(T: Any) -> np.dtype:
    """
    Provide a NumPy type object; `T` may be a plain Python type, a NumPy type
    or dtype, or a type which inherits from the `NPValue` type defined in this
    module.

    Returns
    -------
    type
        Returns a NumPy dtype.
    """
    if isinstance(T, type) and issubclass(T, np.generic):
        # It's important not to call `np.dtype(T)` if T is already a NumPy
        # type, since it prevents generics (e.g. `np.dtype(np.number) is np.float64`)
        # It also triggers a deprecation warning for this reason.
        # T = T.dtype
        pass
    else:
        T = np.dtype(T).type
    return np.dtype(T)

# ###############
# DType


@ABCSerializable.register  # NB: It is not permitted to subclass `np.dtype`
class DType(Serializable):
    """
    Validator also accepts `str` and scalar types (int, np.float32, etc.).
    """
    class Data(SerializedData):
        desc: str
        def encode(dtype): return str(dtype)
        def decode(data): return np.dtype(data.desc)  # Required since we don’t subclass `np.dtype`

    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(type="str")

    @classmethod
    def validate(cls, value, field=None):
        # Following works with any NumPy, but also plain types like int or str
        if isinstance(value, type) and np.issubdtype(value, np.generic):
            return np.dtype(value)
        else:
            return super().validate(value, field=field)

DType.register(np.dtype)
# Because the name 'DType' doesn't match the class we want to make serializable
# (in fact we want to make many classes serializable), we update the registry manually
for type_name in ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'float16', 'float32', 'float64', 'float128',
                  'complex64', 'complex128', 'complex256',
                  'bool_', 'str_'):  # Same list as for NPType
    try:
        dtype = np.dtype(type_name)
    except TypeError:
        # Not all machines define all types; e.g. float128 is not always defined
        pass
    else:
        DType.register(type(dtype))
#         T = type(dtype)
#         if T not in DType._registry:
#             DType.register(T)

# ###############
# NPValue type

def infer_numpy_type_to_cast(nptype, value):
    """
    This function tries to determine which concrete numpy type to use to cast
    `value`, given the desired type `nptype`.
    The challenge is that `nptype` may be an abstract type, so we
    have to work through the possible hierarchy until we find the
    most appropriate concrete type. We only do the most common cases;
    we can extend later to the complete tree.
    """
    # If `nptype` is an NPValue subclass, get the true numpy type
    nptype = getattr(nptype, 'nptype', nptype)
    assert issubclass(nptype, np.generic), f"{nptype} is not a subclass of `np.generic`"
    if nptype is np.generic:
        raise NotImplementedError("Unclear how we should cast np.generic.")
    if issubclass(nptype, np.flexible):  # void, str, unicode
        # Assume concrete type
        return nptype
    elif not issubclass(nptype, np.number):  # bool, object
        assert (nptype is np.bool_ or nptype is np.object_), f"Expected NumPy type 'bool' or 'object'; received '{nptype}'"
        return nptype
    else:  # Number
        if issubclass(nptype, np.integer):
            if nptype is np.integer:
                return np.int_
            else:
                # Assume concrete type
                return nptype
        elif issubclass(nptype, np.inexact):
            if issubclass(nptype, np.complexfloating):
                if nptype is np.complexfloating:
                    return np.complex_
                else:
                    # Assume concrete type
                    return nptype
            elif issubclass(nptype, np.floating):
                if nptype is np.floating:
                    return np.float_
                else:
                    # Assume concrete type
                    return nptype
            else:
                assert nptype is np.inexact, f"Expected NumPy type 'inexact'; received '{nptype}'"
                # We try to guess which type to use between float and complex.
                # We make a rudimentary check for complex, and fall back to float.
                if isinstance(value, complex):
                    return np.complex_
                if isinstance(value, str) and ('j' in value or 'i' in value):
                    return np.complex_
                else:
                    return np.float_
        else:
            assert nptype is np.number, f"Expected NumPy type 'number'; received '{nptype}'"
            # We try to guess which type to use between int, float and complex.
            # We make a rudimentary check for int, complex, and fall back to float.
            if isinstance(value, int):
                return np.int_
            elif isinstance(value, complex):
                return np.complex_
            elif isinstance(value, str):
                if 'j' in value or 'i' in value:
                    return np.complex_
                elif '.' in value:
                    return np.float_
                else:
                    return np.int_
            else:
                return np.float_

def convert_nptype(T):
    """
    Convert a type to a NumPy type (e.g. `float` -> `np.float64`).
    If `T` is already a NumPy type, is it returned unchanged.
    """
    if isinstance(T, type) and issubclass(T, np.generic):
        # It's important not to call `np.dtype(T)` if T is already a NumPy
        # type, since it prevents generics (e.g. `np.dtype(np.number) is np.float64`)
        # It also triggers a deprecation warning for this reason.
        pass
    else:
        T = np.dtype(T).type
    return T

class _NPValueType(Serializable, np.generic):
    nptype = None

    class Data(SerializedData):
        data: Union[float, int, str]
        def encode(val): return val.item()

    # There should be no need for the auto-registration in __init_subclass__,
    # and it can add a lot of cruft to the _registries.
    def __init_subclass__(cls):
        pass

    @classmethod
    def validate(cls, value, field=None):
        # NOTE:
        # We make an exception to always allow casting from a string.
        # This allows for complex values, which may not be converted to numbers
        # by the JSON deserializer and still be represented as strings
        if field is None:
            field = SimpleNamespace(name="")
        if isinstance(value, np.ndarray):
            # Allow scalar arrays
            if value.ndim==0:
                value = value[()]
            else:
                raise ValueError(f"Field {field.name} expects a scalar, not "
                                 f"an array.\nProvided value: {value}.")
        # Don't cast unless necessary
        # np.issubdtype allows specifying abstract dtypes like 'number', 'floating'
        # np.generic ensures isubdtype doesn't let through non-numpy types
        # (like 'float'), or objects which wrap numpy types (like 'ndarray').
        if (isinstance(value, np.generic)
            and np.issubdtype(type(value), cls.nptype)):
            return value
        elif (np.can_cast(value, cls.nptype)
              or np.issubdtype(getattr(value, 'dtype', np.dtype('O')), np.dtype(str))):
            # Exception for strings, as stated above
            nptype = infer_numpy_type_to_cast(cls.nptype, value)
            return nptype(value)
        else:
            raise TypeError(f"Cannot safely cast '{field.name}' type  "
                            f"({type(value)}) to type {cls.nptype}.")
    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     if np.issubdtype(cls.nptype, np.integer):
    #         field_schema.update(type="integer")
    #     else:
    #         field_schema.update(type="number")

class _NPValueMeta(type):
    @cache  # Memoization ensures types are created only once
    def __getitem__(self, nptype):
        nptype=get_dtype(nptype).type
        nptype_str = nptype.__name__
        return type(f'NPValue[{nptype_str}]', (_NPValueType,),
                    {'nptype': nptype})

class NPValue(_NPValueType, metaclass=_NPValueMeta):
    """
    Use this to use a NumPy dtype for type annotation; `pydantic` will
    recognize the type and execute appropriate validation/parsing.

    This may become obsolete, or need to be updated, when NumPy officially
    supports type hints (see https://github.com/numpy/numpy-stubs).

    - `NPValue[T]` specifies an object to be casted with dtype `T`. Any
       expression for which `np.dtype(T)` is valid is accepted.

    .. Note:: **Difference with `DType`.** The annotation `NPValue[np.int8]`
       matches any value of the same type as would be returned by `np.int8`.
       `DType` describes an instance of `dtype` and would match `np.dtype('int8')`,
       but also `np.dtype(float)`, etc.

    Example
    -------
    >>> from pydantic.dataclasses import dataclass
    >>> from scityping.numpy import NPValue
    >>>
    >>> @dataclass
    >>> class Model:
    >>>     x: NPValue[np.float64]
    >>>     y: NPValue[np.int8]

    """
    pass

# Because the name 'NPValue' doesn't match the class we want to make serializable
# (in fact we want to make many classes serializable), we update the registry manually
_NPValueType.register(NPValue)  # Required because we removed __init_subclass__
for type_name in ('int8', 'int16', 'int32', 'int64',
                  'uint8', 'uint16', 'uint32', 'uint64',
                  'float16', 'float32', 'float64', 'float128',
                  'complex64', 'complex128', 'complex256',
                  'bool_', 'str_'):  # Same list as for DType
    try:
        dtype = np.dtype(type_name)
    except TypeError:
        # Not all machines define all types; e.g. float128 is not always defined
        pass
    else:       
        _NPValueType.register(getattr(np, type_name))

# ###
# Array type
#
# Arrays are serialized in two different ways, based on whether it is worth
# compressing them vs simply storing them as a string.

# TODO: Expose an interface for configuring compressors/decompressors
# TODO: Allow different serialization methods:
#       - str(A.tolist())
#       - base64(blosc) (with configurable keywords for blosc)
#       - base64(zlib)
#       - external file

encoders = {"b85": base64.b85encode}
decoders = {"b85": base64.b85decode}
# We use LazyDict to avoid forcing an import of a (possibly not installed) module
compressors = LazyDict(none = lambda s: s,
                       blosc= "blosc.compress",
                       zlib = "zlib.compress")
decompressors = LazyDict(none = lambda s: s, 
                         blosc= "blosc.decompress",
                         zlib = "zlib.decompress")
assert encoders.keys() == decoders.keys(), "Each encoder must have an associated decoder."
assert compressors.keys() == decompressors.keys(), "Each compressor must have an associated decompressor."

# Short arrays we just save as string (more legible and more space efficient).
# Since the string loses type information, we save the dtype as well.
class ListArrayData(SerializedData):
    data : Union[list, StrictFloat, StrictInt]  # Scalar types are to deal with 0-dim arrays
    dtype: DType
    @classmethod
    def encode(cls, array):
        return cls(array.tolist(), array.dtype)
    def decode(data):
        return np.array(data.data, dtype=data.dtype)

# Longer arrays are compressed and converted to base85 encoding, with a short summary
# Arrays of size 100 are around the break-even point for 64-bit floats, blosc, base85
_EncoderType = Literal[tuple(encoders)]         # Workaround because the ability
_CompressionType = Literal[tuple(compressors)]  #   of scityping.Dataclass to resolve
class CompressedArrayData(SerializedData):      #   deferred types is limited
    encoding: _EncoderType
    compression: _CompressionType
    summary: str
    data: bytes
    @classmethod
    def encode(cls, array, compression: Union[str,Tuple[str,...]]=("blosc",),
               encoding="b85", threshold: int=100):
        """
        :param:compression:
           A string matching one of the keys in `scityping.numpy.compressors`.
           May also be a tuple of keys; later keys are used as fallback values,
           if earlier entries are not found (e.g. if blosc is not installed.)
        """
        # Parse arguments
        if compression is None:
            compression = 'none'
        if isinstance(compression, (tuple, list)):
            for comp in compression:
                compressor = compressors.get(comp)
                if compressor:
                    compression = comp  # Value of `compression` is stored with the data
                    break
            else:
                raise ModuleNotFoundError("None of the specified compressors "
                                          f"were found: {compression}.")
        else:
            compressor = compressors[compression]
        encoder = encoders[encoding]
        # Convert array to bytes
        with io.BytesIO() as f:  # Use file object to keep bytes in memory
            np.save(f, array)        # Convert array to plateform-independent bytes  (`tobytes` is not meant for storage)
            v_bytes = f.getvalue()
        # Compress and encode the bytes
        array_encoded = encoder(compressor(v_bytes))
        # Set print threshold to ensure str returns a summary
        with np.printoptions(threshold=threshold):
            array_sum = str(array)
        return cls(encoding, compression, array_sum, array_encoded)
    def decode(data):
        decoder = decoders[data.encoding]
        decompressor = decompressors[data.compression]
        v_bytes = decompressor(decoder(data.data))
        with io.BytesIO(v_bytes) as f:
            decoded_array = np.load(f)
        return decoded_array

class _ArrayType(Serializable, np.ndarray):
    nptype = None   # This must be a type (np.int32, not np.dtype('int32'))
    _ndim = None

    # Dispatches between ListArrayData and CompressedArrayData according to array size
    class Data(SerializedData):
        data: Union[ListArrayData, CompressedArrayData]
        # Class variables; Not yet exposed parameters
        compression = ("blosc", "zlib")  # Fall back on zlib compression if blosc is not available
        encoding = "b85"
        threshold = 100
        @classmethod
        def encode(cls, array, compression=None, encoding=None, threshold=None):
            if array.size <= cls.threshold:
                return {"data": ListArrayData.encode(array)}
            else:
                return {"data": CompressedArrayData.encode(
                    array, compression or cls.compression, encoding or cls.encoding,
                    threshold if threshold is not None else cls.threshold)}
        @staticmethod
        def decode(data):
            return data.data.decode(data.data)

    # There should be no need for the auto-registration in __init_subclass__,
    # and it can add a lot of cruft to the _registries.
    def __init_subclass__(cls):
        pass

    # # TODO: Define schema (should specify that there are two possible forms, and use nptype & _ndim)
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(type ='array',
    #                         items={'type': 'number'})

    @classmethod
    def validate(cls, value, field=None):
        # NB: We don't need to deal with cases where `value` is serialized data:
        #     Serializer.validate + Data.decode will already have converted that
        #     to an array. What we do need to deal with is coercing values to
        #     the required shape and dtype, and raising an error if that is
        #     not possible.
        if field is None:
            field = SimpleNamespace(name="")

        if json_like(value, cls._registry):
            # Rather than growing the whole type hierarchy so `Array[uint,3]`
            # recognizes `Array` as a type it can deserialize, just so we can
            # reuse super().validate, we just do the deserialization here.
            serialized_cls = cls._registry[value[0]]
            data = value[1]
            if not isinstance(data, serialized_cls.Data):
                # breakpoint()
                # data = serialized_cls.Data.validate(data)
                data = serialized_cls.Data(**data)
                validate_dataclass(data, inplace=True)
            value = serialized_cls.Data.decode(data)
            return cls.validate(value)  # Still validate with `cls`: typically `seralized_cls` is *less* specific, like unspecified `Array°

        elif isinstance(value, cls.Data):
            value = cls.Data.decode(value)
            return cls.validate(value)

        elif isinstance(value, np.ndarray):
            # Don't create a new array unless necessary
            if cls._ndim  is not None and value.ndim != cls._ndim:
                raise TypeError(f"{field.name} expects a variable with "
                                f"{cls._ndim} dimensions.")
            # Issubdtype allows specifying abstract numpy types like 'number', 'floating'
            if cls.nptype is None or np.issubdtype(value.dtype, cls.nptype):
                result = value
            elif np.can_cast(value, cls.nptype):
                nptype = infer_numpy_type_to_cast(cls.nptype, value)
                result = value.astype(nptype)
            else:
                raise TypeError(f"Cannot safely cast '{field.name}' (dtype:  "
                                f"{value.dtype}) to type {cls.nptype}.")
        else:
            result = np.array(value)
            # HACK: Since np.array(…) will accept almost anything, we use
            #       heuristics to try to detect when array construction has
            #       probably failed
            if (isinstance(result, (range, Sequence))
                and any(isinstance(x, Iterable)
                        and not isinstance(x, (np.ndarray, str, bytes))
                        for x in result)):
               # Nested iterables should be unwrapped into an n-d array
               # When this fails (if types or depths are inconsistent), then
               # only the outer level is unwrapped.
               raise TypeError(f"Unable to cast {value} to an array.")
            # Check that array matches expected shape and dtype
            if cls._ndim is not None and result.ndim != cls._ndim:
                raise TypeError(
                    f"The dimensionality of the data (dim: {result.ndim}, "
                    f"shape: {result.shape}) does not correspond to the "
                    f"expected of dimensions ({cls._ndim} for '{field.name}').")
            # Issubdtype allows specifying abstract dtypes like 'number', 'floating'
            if cls.nptype is None or np.issubdtype(result.dtype, cls.nptype):
                pass
            else:
                # Inferring the nptype first avoids warnings if cls.nptype is an abstract type, like np.inexact
                nptype = infer_numpy_type_to_cast(cls.nptype, result)
                if np.can_cast(result, nptype, casting="safe"):
                    result = result.astype(nptype)
                else:
                    raise TypeError(f"Cannot safely cast '{field.name}' (dtype:  "
                                    f"{result.dtype}) to an array of type {cls.nptype}.")
        return result

class _ArrayMeta(type):
    @cache  # Memoization ensures types are created only once
    def __getitem__(self, args):
        if isinstance(args, tuple):
            T = args[0]
            ndim = args[1] if len(args) > 1 else None
            extraargs = args[2:]  # For catching errors only
        else:
            T = args
            ndim = None
            extraargs = []
        if isinstance(T, np.dtype):
            T = T.type
        elif isinstance(T, str):
            T = np.dtype(T).type
        if (not isinstance(T, type) or len(extraargs) > 0
            or not isinstance(ndim, (int, type(None)))):
            # if isinstance(args, (tuple, list)):
            #     argstr = ', '.join((str(a) for a in args))
            # else:
            #     argstr = str(args)
            raise TypeError(
                "`Array` must be specified as either `Array[T]`"
                "or `Array[T, n], where `T` is a type and `n` is an int. "
                f"(received: {args}]).")
        nptype=convert_nptype(T)
        # specifier = str(nptype)
        specifier = nptype.__name__
        if ndim is not None:
            specifier += f",{ndim}"
        # There should be no need for the auto-registration in __init_subclass__,
        # and it can add a lot of cruft to the _registries.
        return type(f'Array[{specifier}]', (_ArrayType,),
                    {'nptype': nptype, '_ndim': ndim})

# NB: We keep Array and _ArrayType separate, so that only Array has the _ArrayMeta metaclass
class Array(_ArrayType, metaclass=_ArrayMeta):
    """
    Use this to specify a NumPy array type annotation; `pydantic` will
    recognize the type and execute appropriate validation/parsing.

    This may become obsolete, or need to be updated, when NumPy officially
    supports type hints (see https://github.com/numpy/numpy-stubs).

    - `Array[T]` specifies an array with dtype `T`. Any expression for which
      `np.dtype(T)` is valid is accepted.
    - `Array[T,n]` specifies an array with dtype `T`, that must have exactly
      `n` dimensions.

    Example
    -------
    >>> from pydantic.dataclasses import dataclass
    >>> from scityping import Array
    >>>
    >>> @dataclass
    >>> class Model:
    >>>     x: Array[np.float64]      # Array of 64-bit floats, any number of dimensions
    >>>     v: Array['float64', 1]    # 1-D array of 64-bit floats
    """
    pass

_ArrayType.register(Array)  # Required because we removed __init_subclass__
_ArrayType.register(np.ndarray)

# ####
# NumPy random generators
#
# Generators are containers around a BitGenerator; it's the state of
# BitGenerator that we need to save. Default BitGenerator is 'PCG64'.
# State is a dict with two required fields: 'bit_generator' and 'state',
# plus 1-3 extra fields depending on the generator.
# 'state' field is itself a dictionary, which may contain arrays of
# type uint32 or uint64

RNGStateDict = Dict[str, Union[int, Array[np.unsignedinteger,1]]]

class RNGenerator(Serializable, np.random.Generator):
    class Data(SerializedData):
        bit_generator: str                       # Philox, PCG64, SFG64, MT19937  
        state        : RNGStateDict              # Philox, PCG64, SFG64, MT19937
        has_uint32   : Optional[int]=None        # Philox, PCG64, SFG64
        uinteger     : Optional[int]=None        # Philox, PCG64, SFG64
        buffer       : Optional[Array[np.uint64,1]]=None  # Philox
        buffer_pos   : Optional[int]=None        # Philox
            # Pydantic will recursively encode state entries, and use Array's
            # reduce when needed
        def encode(rng):
            return rng.bit_generator.state
                # Since this is a dict, RNGenerator.reduce will pass by keyword to `Data`
        def decode(data: 'RNGenerator.Data') -> np.random.Generator:
            bg = getattr(np.random, data.bit_generator)()
            bg.state = asdict(data)
            return np.random.Generator(bg)
    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(type='array',
    #                         items=[{'type': 'string',
    #                                 'type': 'object'}])

NPGenerator = RNGenerator  # Previous name
Generator = RNGenerator  # To match the name in NumPy

NPGenerator.register(np.random.Generator)

class RandomState(Serializable, np.random.RandomState):
    """Pydantic typing support for the legacy RandomState object.

    .. Note:: Currently tested with the MT19937 generator.
    """
    class Data(SerializedData):
        state: Tuple[str, Array[np.uint64], int, int, float]  # name: str, state: array, int, int, float
        def encode(random_state): return {"state": random_state.get_state()}  # Simply returning tuple would be interpreted as arguments to Data
        def decode(data):
            rs = np.random.RandomState()
            rs.set_state(data.state)
            return rs

    # # TODO
    # @classmethod
    # def __modify_schema__(cls, field_schema):
    #     field_schema.update(
    #     type='array',
    #     items=[{'type': 'string'},
    #            {'type': 'array',
    #             'items': [{'type': 'array', 'items': {'type': 'integer'}},
    #                       {'type': 'integer'},
    #                       {'type': 'integer'},
    #                       {'type': 'number'}]}
    #           ])

# ####
# NumPy SeedSequence
#
# NOTE: A serialized `SeedSequence` always has a fixed entropy, even when
#    it was created with `entropy=None`. If you need to serialize a `SeedSequence`
#    which returns different values every time it is loaded, you may serialize
#    the arguments and recreate the `SeedSequence` in your code. But also
#    consider that serializing a non-deterministic object is an anti-pattern.

class SeedSequence(Serializable, np.random.SeedSequence):
    class Data(SerializedData):
        entropy           : Union[int,List[int]]
        spawn_key         : Tuple[int,...]
        pool_size         : int
        n_children_spawned: int
        def encode(seedseq) -> SeedSequence.Data:
            return (seedseq.entropy, seedseq.spawn_key,
                    seedseq.pool_size, seedseq.n_children_spawned)

