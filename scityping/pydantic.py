"""
Tie-ins for pydantic which explicitely depend on pydantic types.
Provides:

- An scityping-aware "default" callable for `json.dump`, which allows it to
  serialize `Serializable` types.
- Drop-in replacements for `pydantic.BaseModel` and
  `pydantic.dataclasses.dataclass`, which by default uses the scityping JSON
  encoder provided by this package.
"""
from typing import TYPE_CHECKING, Any, Type
from functools import partial
from dataclasses import is_dataclass
from importlib.metadata import version
from pydantic import BaseModel as PydanticBaseModel
from pydantic.main import (ModelMetaclass as PydanticModelMetaclass,
                           ValidationError as PydanticValidationError)
from pydantic.generics import GenericModel as PydanticGenericModel
from pydantic.error_wrappers import display_errors
from pydantic.dataclasses import (dataclass as pydantic_dataclass,
                                  _validate_dataclass as _pydantic_validate_dataclass)
from .base import Serializable, json_like, Dataclass
from .json import scityping_encoder

if TYPE_CHECKING:
    from pydantinc.typing import ReprArgs

class ValidationError(PydanticValidationError):
    """
    This wraps ValidationError, and redefines the methods using `__name__`
    so they use `__qualname__` instead.
    """
    def __init__(self, orig_exception: PydanticValidationError):
        super().__init__(orig_exception.raw_errors, orig_exception.model)
        self._error_cache = orig_exception._error_cache

    def __str__(self) -> str:
        errors = self.errors()
        no_errors = len(errors)
        return (
            f'{no_errors} validation error{"" if no_errors == 1 else "s"} for {self.model.__qualname__}\n'
            f'{display_errors(errors)}'
        )

    def __repr_args__(self) -> 'ReprArgs':
        return [('model', self.model.__qualname__), ('errors', self.errors())]


def remove_overridden_validators(model: PydanticBaseModel) -> PydanticBaseModel:
    """
    Currently a Pydantic bug prevents subclasses from overriding root validators.
    (see https://github.com/samuelcolvin/pydantic/issues/1895)
    This function inspects a Pydantic model and removes overriden
    root validators based of their `__name__`.
    Assumes that the latest entries in `__pre_root_validators__` and
    `__post_root_validators__` are earliest in the MRO, which seems to be
    the case.
    """
    model.__pre_root_validators__ = list(
        {validator.__name__: validator
         for validator in model.__pre_root_validators__
         }.values())
    model.__post_root_validators__ = list(
        {validator.__name__: (skip_on_failure, validator)
         for skip_on_failure, validator in model.__post_root_validators__
         }.values())
    return model

class ModelMetaclass(PydanticModelMetaclass):
    """
    Use this as the metaclass for BaseModel types, to allow them

    1. to find serializers defined in Serializable;
    2. to allow their subclasses to override root validators,
       (patch for https://github.com/samuelcolvin/pydantic/issues/1895)
       (Only applied if Pydantic version is ⩽ 1.8)
    """
    def __new__(mcs, name, bases, namespace, **kwargs):
        ann = namespace.get("__annotations__", {})
        for field, T in ann.items():
            if is_dataclass(T) and not isinstance(T, Serializable):  # Serializable subclasses already have an __get_validators which yields cls.validate
                orig_get_validators = getattr(T, "__get_validators__", None)
                if not orig_get_validators:
                    try:
                        T.__get_validators__ = _get_dataclass_validator(T)
                    except AttributeError:
                        logger.error(
                            "Cannot add attribute '__get_validators__' to the "
                            f"type `{T.__qualname__}`: the class `{name}` will "
                            "not be deserializable. This can happen if the class "
                            "was defined with '__slots__'.")
        obj = super().__new__(mcs, name, bases, namespace, **kwargs)
        # Wrap the reduce with scityping_encoder
        encoder = partial(scityping_encoder, base_encoder=obj.__json_encoder__)
        obj.__json_encoder__ = staticmethod(encoder)
        # If Pydantic version is ⩽ 1.8, apply patch to allow their subclasses to override root validators.
        _pydantic_version = tuple(int(i) for i in version("pydantic").split("."))
        if _pydantic_version <= (1, 8):
            obj = remove_overridden_validators(obj)
        return obj

class BaseModel(PydanticBaseModel, metaclass=ModelMetaclass):
    pass

class GenericModel(PydanticGenericModel, BaseModel):
    pass

def _get_dataclass_validator(T: type):
    def _dataclass_type_match(value):
        if not isinstance(value, T):
            raise TypeError(f"Value is not of type `{T.__qualname__}`")
        return value
    def _get_validators():
        yield _dataclass_deserializer
        yield _dataclass_type_match
    return _get_validators

def _dataclass_deserializer(value):
    if json_like(value, Dataclass._registry):
        return Dataclass.validate(value)
    else:
        return value

# C.f. pydantic.dataclasses
def _validate_serializable_dataclass(cls: Type["DataclassT"], v: Any) -> "DataclassT":
    if json_like(v, cls._registry):
        return cls.validate(v)
    else:
        return _pydantic_validate_dataclass(cls, v)

# Hard-coded special-case support for stdlib dataclasses
def _validate_plain_dataclass(cls: Type["DataclassT"], v: Any) -> "DataclassT":
    if json_like(v, Dataclass._registry):
        return Dataclass.validate(v)
    else:
        return _pydantic_validate_dataclass(cls, v)

def dataclass(_cls=None, **kwargs):
    ## Secondary code path: decorator with arguments ##
    if _cls is None:
        return partial(dataclass, **kwargs)

    ## Primary code path: decorator without arguments ##

    # Validation in a Pydantic dataclass is performed by storing a BaseModel
    # in the private attribute __pydantic_model__. Additional dunder methods are
    # added to the dataclass: The validation logic for the class itself (i.e.
    # what is used when the class is part of a BaseModel) is attached to __validate__.
    # First we create the dataclass, then patch the validation & serialization
    # logic so they recognize Serializable types
    dclass = pydantic_dataclass(_cls, **kwargs)

    # Validation: Patch __validate__ to recogize a Serialized `self`
    #    Note that any Serializable *arguments* to the dataclass will already
    #    be correctly deserialized. What we need to catch is the case where
    #    `self` is a dataclass, and we are instantiating it from serialized data.
    #    Therefore we only need to wrap __validate__, not __pydantic_validate_values__
    if issubclass(dclass, Serializable):
        setattr(dclass, "__validate__", classmethod(_validate_serializable_dataclass))
    else:
        # Hard-coded special-case support for stdlib dataclasses
        setattr(dclass, "__validate__", classmethod(_validate_plain_dataclass))
    
    # JSON encoder: As in `ModelMetaclass`, we patch the __json_encoder__ attribute
    obj = dclass.__pydantic_model__
    encoder = partial(scityping_encoder, base_encoder=obj.__json_encoder__)
    obj.__json_encoder__ = staticmethod(encoder)

    # Return
    return dclass
