"""
Tie-ins for pydantic which explicitely depend on pydantic types.
Provides:
- An scityping-aware "default" callable for `json.dump`, which allows it to
  serialize `Serializable` types.
- Drop-in replacements for `pydantic.BaseModel` and
  `pydantic.dataclasses.dataclass`, which by default uses the scityping JSON
  encoder provided by this package.
"""
from typing import Any, Type
from functools import partial
from pydantic import BaseModel as PydanticBaseModel
from pydantic.generics import GenericModel as PydanticGenericModel
from pydantic.main import (ModelMetaclass as PydanticModelMetaclass,
                           ValidationError as PydanticValidationError)
from pydantic.json import pydantic_encoder
from pydantic.dataclasses import (dataclass as pydantic_dataclass,
                                  _validate_dataclass as _pydantic_validate_dataclass)
from .base import ABCSerializable, Serializable, json_like

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


# Based off pydantic.json.custom_pydantic_encoder
def scityping_encoder(obj: Any, base_encoder=pydantic_encoder) -> Any:
    if isinstance(obj, ABCSerializable):
        try:
            # NB: The function pydantic.json.pydantic_encoder checks if the argument
            #     is a BaseModel or dataclass, and if so, calls respectively .dict()
            #     or asdict() to convert it to a dictionary.
            #     *These are recursive calls*, which means that they reduce
            #     every value inside them using only that function’s machinery:
            #     nested calls to `pydantic_encoder` are *not* made, and in
            #     particular `asdict` will ignore any custom encoders – it even
            #     calls `deep_copy` on its contents.
            # We emulate pydantic..pydantic_encoder here and call a recursive
            # function to also reduce any nested Serializable values within `obj`.
            # If we only reduce the `obj` but not its attributes,
            # pydantic..pydantic_encoder may bypass `Serializable.reduce` for
            # Serializable attributes.
            return Serializable.deep_reduce(obj)
        except PydanticValidationError as e:
            # Pydantic's ValidationError uses obj.__name__ it its error message.
            # With our SciTyping pattern, this results in all error message
            # displaying `Data` as the object, which isn't very useful.
            # To improve the error message, we modify the model attached to
            # the exception so that its __name__ is actually __qualname__.
            # This monkey patching isin't especially clean, but it should be
            # innocuous: after all, we've already aborted code execution.
            if e.model.__name__ == "Data":
                e.model.__name__ = e.model.__qualname__
            raise
    else:
        return base_encoder(obj)

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
    a) to find serializers defined in Serializable
    b) to allow their subclasses to override root validators
       (patch for https://github.com/samuelcolvin/pydantic/issues/1895)
    """
    def __new__(mcs, name, bases, namespace, **kwargs):
        obj = super().__new__(mcs, name, bases, namespace, **kwargs)
        # Wrap the reduce with scityping_encoder
        encoder = partial(scityping_encoder, base_encoder=obj.__json_encoder__)
        obj.__json_encoder__ = staticmethod(encoder)
        # Apply patch to allow their subclasses to override root validators.
        return remove_overridden_validators(obj)

class BaseModel(PydanticBaseModel, metaclass=ModelMetaclass):
    pass

class GenericModel(PydanticGenericModel, BaseModel):
    pass

# C.f. pydantic.dataclasses
def _validate_serializable_dataclass(cls: Type["DataclassT"], v: Any) -> "DataclassT":
    if json_like(v, cls._registry):
        return cls.validate(v)
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

    # Validation: Patch __pydantic_validate_values__ to recogize a Serialized `self`
    #    Note that any Serializable *arguments* to the dataclass will already
    #    be correctly deserialized. What we need to catch is the case where
    #    `self` is both a dataclass and Serializable, and we are instantiating
    #    it from serialized data.
    #    Therefore we only need to wrap __validate__, not __pydantic_validate_values__
    if issubclass(dclass, Serializable):
        setattr(dclass, "__validate__", classmethod(_validate_serializable_dataclass))
    
    # JSON encoder: As in `ModelMetaclass`, we patch the __json_encoder__ attribute
    obj = dclass.__pydantic_model__
    encoder = partial(scityping_encoder, base_encoder=obj.__json_encoder__)
    obj.__json_encoder__ = staticmethod(encoder)

    # Return
    return dclass
