"""
Tie-ins for pydantic which explicitely depend on pydantic types.
Provides:
- An extensible JSON encoder, which works with `base_types.Serializable` to
  allow modules to define new types which work with smttask's serialization
  infrastructure.
- Drop-in replacements for `pydantic.BaseModel` and
  `pydantic.dataclasses.dataclass`, which by default uses the extensible JSON
  encoder provided by this package.
"""
from typing import Any
from functools import partial
from pydantic import BaseModel as PydanticBaseModel
from pydantic.generics import GenericModel as PydanticGenericModel
from pydantic.main import ModelMetaclass as PydanticModelMetaclass
from pydantic.json import custom_pydantic_encoder
from pydantic.dataclasses import dataclass as pydantic_dataclass
from .base import ABCSerializable, Serializable

# Based off pydantic.json.custom_pydantic_encoder
def extensible_encoder(obj: Any, base_encoder) -> Any:
    if isinstance(obj, ABCSerializable):
        return Serializable.json_encoder(obj)
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
        # Wrap the json_encoder with extensible_encoder
        encoder = partial(extensible_encoder, base_encoder=obj.__json_encoder__)
        obj.__json_encoder__ = staticmethod(encoder)
        # Apply patch to allow their subclasses to override root validators.
        return remove_overridden_validators(obj)

class BaseModel(PydanticBaseModel, metaclass=ModelMetaclass):
    pass

class GenericModel(PydanticGenericModel, BaseModel):
    pass

# C.f. pydantic.dataclasses
def dataclass(_cls, **kwargs):
    # Validation in a Pydantic dataclass is performed by storing a BaseModel
    # in the private attribute __pydantic_model__. Thus as in `ModelMetaclass`,
    # we can first create the model, then the JSON encoder
    dclass = pydantic_dataclass(_cls, **kwargs)
    obj = dclass.__pydantic_model__
    encoder = partial(extensible_encoder, base_encoder=obj.__json_encoder__)
    obj.__json_encoder__ = staticmethod(encoder)
    # Return
    return dclass
