"""
Manifest
========

Types:
------
  + Torch types:
    - TorchTensor
    - TorchModule  (validation only)
    - TorchGenerator

JSON encoders
-------------
  + Torch types:
    - torch.Tensor
    - torch.nn.module (provided as a pair of methods: torch_module_state_encoder, torch_module_state_decoder)
    - torch.Generator
"""

from __future__ import annotations

from typing import Tuple, Dict

from .base import Serializable, ABCSerializable, json_like
from .base_types import SerializedData
from .numpy import Array

import numpy  # There are sometimes issues when torch is imported before numpy
import torch
import torch.nn as nn

# TODO: Allow indexing as with Array, to specify dtype & ndim
class TorchTensor(Serializable, torch.Tensor):
    class Data(SerializedData):
        data: Array
        def encode(tensor): return tensor.cpu().detach().numpy()
        def decode(data): return torch.tensor(data.data)

Tensor = TorchTensor

type_to_register = torch.Tensor
cls = TorchTensor
ABCSerializable.register(type_to_register)
ABCSerializable._base_types[cls] = type_to_register
for C in cls.mro():
    if issubclass(C, Serializable):
        C._registry[type_to_register] = cls

class TorchModule(nn.Module):
    """
    Validation only type; for serializing a PyTorch model, see `scityping.ml.TorchModel`.
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if not isinstance(v, nn.Module):
            raise TypeError("Expected a torch.nn.Module; received "
                            f"'{v}' (type: {type(v)})")
        return v

Module = TorchModule

TorchModuleState = Tuple[str, Dict[str, Array]]

# TODO: Package into a TorchModuleState class
def torch_module_state_decoder(v: Tuple[str, Dict[str,str]]):
    """
    Decode a torch module state serialized with `torch_module_state_reduce`.
    """
    if not json_like(v, "TorchModuleState"):
        raise TypeError("Argument is not a serialized torch state. "
                        f"Received: {v}.")
    encoded_state = v[1]
    # State is encoded as serialized arrays, so for each we have to:
    # 1. Deserialize the array  2. Convert to a pytorch tensor
    state = {param: torch.tensor(array) for param, array in encoded_state.items()}
    return state

def torch_module_state_encoder(
      v: nn.Module, compression: str='blosc', encoding: str='b85'
    ) -> Tuple[str, Dict[str,str]]:
    """
    Encode to PyTorch model state into an ASCII string, which can
    be included in a JSON file.

    .. Note:: This only encodes the model _state_ (i.e. the value
       from `v.state_dict()`). To reconstruct the model, one also
       needs to store the code, or some other means to reconstruct
       the architecture.

    Parameters
    ----------
    v: PyTorch model
    compression: 'blosc' | 'none'
       Which routine to use to compress bytes, if any.
    encoding: 'b85'
       Which ASCII encoding to use. Currently only 'b85' is supported.
       The `base64` module is used for this encoding.

    Returns
    -------
    ``("TorchModuleState", <data>)``
    where `<data>` is a dictionary with four fields:
       - encoding: Value of `encoding`
       - compression: Value of `compression`
       - description: Free-form, human-readable summary of the model
           (obtained by ``str(v)``)
       - state: The encoded model state.

    """
    if not isinstance(v, nn.Module):
        raise TypeError("This JSON encoder is only intended for PyTorch "
                        f"modules (received value of type {type(v)}).")
    state = {param: tensor.cpu().detach().numpy()
             for param, tensor in v.state_dict().items()}

    return ("TorchModuleState", state)


class TorchGenerator(Serializable, torch.Generator):
    """
    Pydantic-aware torch Generator.
    Since random number generators are platform dependent, serializing-
    deserializing from one machine to another, or from CPU to GPU, is likely
    to produce different results. However on the same machine and device,
    results should be consistent.
    """
    class Data(SerializedData):
        device: str
        state: TorchTensor
        def encode(gen): return gen.device.type, gen.get_state()
        def decode(data):
            try:
                gen = torch.Generator(data.device)
            except RuntimeError:
                gen = torch.Generator()
                logger.error(
                    "Unable to instantiate a torch Generator on device "
                    f"'{data.device}'. Used the default '{gen.device}' instead.")
            # NB: It seems to be a bug in PyTorch that even for a torch generator
            #     located on GPU, `set_state` only accepts arguments located on CPU.
            #     Otherwise, presumably the error message
            #     – "RNG state must be a torch.ByteTensor" – would be more useful.
            gen.set_state(data.state.to("cpu"))
            return gen

Generator = TorchGenerator

type_to_register = torch.Generator
cls = TorchGenerator
ABCSerializable.register(type_to_register)
ABCSerializable._base_types[cls] = type_to_register
for C in cls.mro():
    if issubclass(C, Serializable):
        C._registry[type_to_register] = cls
