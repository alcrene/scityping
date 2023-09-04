from typing import Any
from collections.abc import Callable
from dataclasses import is_dataclass
from .base import ABCSerializable, Serializable, Dataclass
from .functions import serialize_function

try:
    from pydantic.main import ValidationError
except ModuleNotFoundError:
    # If pydantic can’t be imported, it is fine to create a dummy class which
    # will never match
    class ValidationError(RuntimeError):
        pass

# Based on pydantic.json.custom_pydantic_encoder
def scityping_encoder(obj: Any, base_encoder=None) -> Any:
    """
    This function can be used as the `default` argument to `json.dump`, to
    add support for `Serializable` types.
    """
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
        except ValidationError as e:
            # Pydantic's ValidationError uses obj.__name__ it its error message.
            # With our SciTyping pattern, this results in all error message
            # displaying `Data` as the object, which isn't very useful.
            # To improve the error message, we modify the model attached to
            # the exception so that its __name__ is actually __qualname__.
            # This monkey patching isn't especially clean, but it should be
            # innocuous: after all, we've already aborted code execution.
            if e.model.__name__ == "Data":
                e.model.__name__ = e.model.__qualname__
            raise e
    elif is_dataclass(obj) and not isinstance(obj, type):
        # Dataclasses need to be special cased because there is no
        # base dataclass, so we can't identify them with `isinstance`
        # They are worth special-casing because they are one of the basic types
        # used for packaging data, in particular for our nested Data classes
        return Dataclass.deep_reduce(obj)
    elif isinstance(obj, Callable):        # NB: We don't want to do this within 'Data.encode', because sometimes we
        try:
            return serialize_function(obj)     #     use 'encode' without going all the way to a JSON string
        except TypeError as e:
            # Things like callable Pydantic models may not be serializable with the function serializer,
            # but still serializable with the base encoder.
            # (NB: We don’t try the base encoder first, because we want to give
            # the function serializer priority.)
            if base_encoder:
                return base_encoder(obj)
            else:
                raise e
    elif base_encoder:
        return base_encoder(obj)
    else:
        raise TypeError

