from __future__ import annotations

import sys
import importlib
import inspect
import operator
import functools

from typing import Union, Callable, List, Sequence, _Final
from types import new_class

from .base import Serializable
from .base_types import SerializedData, Number
from .numpy import Array  # TODO: Make numpy import optional

PlainArg = Union[Number, str, Array]

__all__ = ["PlainArg", "PureFunction", "PartialPureFunction", "CompositePureFunction"]

# TODO: Instead of "trust_all_inputs", use a whitelist of safe module (as in scityping.scipy.Distribution)
# TODO: Serializing partial(PureFunction(g)) should be the same as PartialPureFunction(partial(g)), for consistent hashes

class PureFunctionMeta(type):
    _instantiated_types = {}
    def __getitem__(cls, args):
        """
        Returns a subclass of `PureFunction`. Args may consist of
        - The callable type (in the same format as `~typing.Callable`).
        - Module names. These are used to define namespaces which should be
          imported into the local namespace during deserialization.
        - Both a callable type and module names.
        
        .. Note::
           Types cannot be specified as strings – string arguments are assumed
           to be module names.
        """
        # Parse the arguments
        callableT = {'inT': None, 'outT': None}
        modules = []
        for a in args:
            if isinstance(a, str):
                modules.append(a)
            elif inspect.ismodule(a):
                for nm, m in sys.modules.items():
                    if m is a:
                        modules.append(nm)
                        break
                else:
                    raise AssertionError(f"Module {a} not found in `sys.modules`.")
            elif isinstance(a, list):
                if callableT['inT'] is not None:
                    raise TypeError("Only one input type argument may be specified to"
                                     f"`PureFunction`. Received {callableT['inT']} and {a}.")
                callableT['inT'] = a
            elif isinstance(a, (_Final, type)) or a is None:
                if callableT['outT'] is not None:
                    raise TypeError("Only one output type argument may be specified to"
                                     f"`PureFunction`. Received {callableT} and {a}.")
                if a is None:
                    a = type(None)  # This is what Callable does automatically anyway, and it allows us to check below that either both of inT, outT were passed, or neither
                callableT['outT'] = a
            else:
                raise TypeError("Arguments to the `PureFunction` type can "
                                "consist of zero or one type and zero or more "
                                f"module names. Received {a}, which is of type "
                                f"type {type(a)}.")
        # Treat the callable type, if present
        if (callableT['inT'] is None) != (callableT['outT'] is None):
            raise TypeError("Either both the input and output type of a "
                            "PureFunction must be specified, or neither.")
        if callableT['inT']:
            assert callableT['outT'] is not None
            baseT = Callable[callableT['inT'], callableT['outT']]
            argstr = f"{callableT['inT']}, {callableT['outT']}"
        else:
            baseT = Callable
            argstr = ""
        # Treat the module names, if present
        if modules:
            if argstr:
                argstr += ", "
            argstr += ", ".join(modules)
        # Check if this PureFunction has already been created, and if not, do so
        key = (cls, baseT, tuple(modules))
        if key not in cls._instantiated_types:
            PureFunctionSubtype = new_class(
                f'{cls.__name__}[{argstr}]', (cls,))
            cls._instantiated_types[key] = PureFunctionSubtype
            PureFunctionSubtype.modules = cls.modules + modules
        # Return the PureFunction type
        return cls._instantiated_types[key]

class PureFunction(Serializable, metaclass=PureFunctionMeta):
    """
    A Pydantic-compatible function type, which supports deserialization.
    A “pure function” is one with no side-effects, and which is entirely
    determined by its inputs.

    Accepts also partial functions, in which case an instance of the subclass
    `PartialPureFunction` is returned.

    .. Warning:: Deserializing functions is necessarily fragile, since there
       is no way of guaranteeing that they are truly pure.
       When using a `PureFunction` type, always take extra care that the inputs
       are sane.

    .. Note:: Functions are deserialized without the scope in which they
       were created.

    .. Hint:: If ``f`` is meant to be a `PureFunction`, but defined as::

       >>> import math
       >>> def f(x):
       >>>   return math.sqrt(x)

       then it has dependency on ``math`` which is outside its scope, and is
       thus impure. It can be made pure by putting the import inside the
       function::

       >>> def f(x):
       >>>   import math
       >>>   return math.sqrt(x)

    .. Note:: Like `Callable`, `PureFunction` allows to specify the type
       within brackets: ``PureFunction[[arg types], return y]``. However the
       returned type doesn't support type-checking.

    .. WIP: One or more modules can be specified to provide definitions for
       deserializing the file, but these modules are not serialized with the
       function.
    """
    modules = []  # Use this to list modules that should be imported into
                  # the global namespace before deserializing the function
    # subtypes= {}  # Dictionary of {JSON label: deserializer} pairs.
    #               # Use this to define additional PureFunction subtypes
    #               # for deserialization.
    #               # JSON label is the string stored as first entry in the
    #               # serialized tuple, indicating the type.
    #               # `deserializer` should be a function (usually the type's
    #               # `deserialize` method) taking the serialized value and
    #               # returning the PureFunction instance.
    # Instance variable
    func: Callable

    class Data(SerializedData):
        func: Union[str,PureFunction]
        def encode(purefunc: PureFunction) -> str:
            func = purefunc.func
            if isinstance(func, PureFunction):
                return func
            else:
                return serialize_function(purefunc.func)
        def decode(data: PureFunction.Data) -> PureFunction:
            # return PureFunction.deserialize(data.func)
            cls = PureFunction
            modules = [importlib.import_module(m_name) for m_name in cls.modules]
            global_ns = {k:v for m in modules
                             for k,v in m.__dict__.items()}
            # Since decorators are serialized with the function, we should at
            # least make the decorators in this module available.
            local_ns = {'PureFunction': PureFunction,
                        'PartialPureFunction': PartialPureFunction,
                        'CompositePureFunction': CompositePureFunction}
            pure_func = deserialize_function(data.func, global_ns, local_ns)
            # It is possible for a function to be serialized with a decorator
            # which returns a PureFunction, or even a subclass of PureFunction
            # In such a case, casting as PureFunction may be destructive, and
            # is at best useless
            if not isinstance(pure_func, cls):
                pure_func = cls(pure_func)
            return pure_func

    def __new__(cls, func=None):
        # func=None allowed to not break __reduce__ (due to metaclass)
        # – inside a __reduce__, it's fine because __reduce__ will fill __dict__ after creating the empty object
        if isinstance(func, functools.partial) and not issubclass(cls, PartialPureFunction):
            # Redirect to PartialPureFunction constructor
            # FIXME: What if we get here from CompositePureFunction.__new__
            try:
                Partial = cls.__dict__['Partial']
            except KeyError:
                raise TypeError(f"{func} is a partial functional but '{cls}' "
                                "does not define a partial variant.")
            else:
                return Partial(func)
        # if cls is PureFunction and isinstance(func, functools.partial):
        #     # Redirect to PartialPureFunction constructor
        #     # FIXME: What if we get here from CompositePureFunction.__new__
        #     return PartialPureFunction(func)
        return super().__new__(cls)
    def __init__(self, func):
        if hasattr(self, 'func'):
            # This is our second pass through __init__, probably b/c of __new__redirect
            assert hasattr(self, '__signature__')
            return
        self.func = func
        # Copy attributes like __name__, __module__, ...
        functools.update_wrapper(self, func)
        self.__signature__ = inspect.signature(func)
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    ## Function arithmetic ##
    def __abs__(self):
        return CompositePureFunction(operator.abs, self)
    def __neg__(self):
        return CompositePureFunction(operator.neg, self)
    def __pos__(self):
        return CompositePureFunction(operator.pos, self)
    def __add__(self, other):
        if other == 0:  # Allows using sum([PureFn]) without creating unnecessary Composite functions
            return self
        return CompositePureFunction(operator.add, self, other)
    def __radd__(self, other):
        if other == 0:  # Idem
            return self
        return CompositePureFunction(operator.add, other, self)
    def __sub__(self, other):
        if other == 0:  # Idem
            return self
        return CompositePureFunction(operator.sub, self, other)
    def __rsub__(self, other):
        return CompositePureFunction(operator.sub, other, self)
    def __mul__(self, other):
        return CompositePureFunction(operator.mul, self, other)
    def __rmul__(self, other):
        return CompositePureFunction(operator.mul, other, self)
    def __truediv__(self, other):
        return CompositePureFunction(operator.truediv, self, other)
    def __rtruediv__(self, other):
        return CompositePureFunction(operator.truediv, other, self)
    def __pow__(self, other):
        return CompositePureFunction(operator.pow, self, other)

    ## Serialization / deserialization ##
    # The attribute '__func_src__', if it exists,
    # is required for deserialization. This attribute is added by
    # `deserialize_function` when it deserializes a function string.
    # We want it to be attached to the underlying function, to be sure
    # the serializer can find it
    @property
    def __func_src__(self):
        return self.func.__func_src__
    @__func_src__.setter
    def __func_src__(self, value):
        self.func.__func_src__ = value

    # @classmethod
    # def deserialize(cls, value) -> PureFunction:
    #     if isinstance(value, PureFunction):
    #         pure_func = value
    #     elif isinstance(value, Callable):
    #         pure_func = PureFunction(value)
    #     elif isinstance(value, str):
    #         modules = [importlib.import_module(m_name) for m_name in cls.modules]
    #         global_ns = {k:v for m in modules
    #                          for k,v in m.__dict__.items()}
    #         # Since decorators are serialized with the function, we should at
    #         # least make the decorators in this module available.
    #         local_ns = {'PureFunction': PureFunction,
    #                     'PartialPureFunction': PartialPureFunction,
    #                     'CompositePureFunction': CompositePureFunction}
    #         pure_func = deserialize_function(
    #             value, global_ns, local_ns)
    #         # It is possible for a function to be serialized with a decorator
    #         # which returns a PureFunction, or even a subclass of PureFunction
    #         # In such a case, casting as PureFunction may be destructive, and
    #         # is at best useless
    #         if not isinstance(pure_func, cls):
    #             pure_func = cls(pure_func)
    #     # elif (isinstance(value, Sequence) and len(value) > 0):
    #     #     label = value[0]
    #     #     if label == "PartialPureFunction":
    #     #         pure_func = PartialPureFunction.deserialize(value)
    #     #     elif label == "CompositePureFunction":
    #     #         pure_func = CompositePureFunction.deserialize(value)
    #     #     elif label in cls.subtypes:
    #     #         pure_func = cls.subtypes[label](value)
    #     #     else:
    #     #         cls.raise_validation_error(value)
    #     else:
    #         cls.raise_validation_error(value)
    #     return pure_func

    # TODO: Add arg so PureFunction subtype can be specified in error message
    @classmethod
    def raise_validation_error(cls, value):
        raise TypeError("PureFunction can be instantiated from either "
                        "a callable, "
                        "a Sequence `([PureFunction subtype name], func, bound_values)`, "
                        "or a string. "
                        f"Received {value} (type: {type(value)}).")

    # @staticmethod
    # def serialize(v) -> str:
    #     if isinstance(v, PartialPureFunction):
    #         return PartialPureFunction.serialize(v)
    #     elif isinstance(v, CompositePureFunction):
    #         return CompositePureFunction.serialize(v)
    #     elif isinstance(v, PureFunction):
    #         f = v.func
    #     elif isinstance(v, Callable):
    #         f = v
    #     else:
    #         raise TypeError("`PureFunction.serialize` only accepts "
    #                         f"functions as arguments. Received {type(v)}.")
    #     return serialize_function(f)

# If pydantic is available, it is used for the Data classes, and we need to resolve the circular dependency
pydantic_model = getattr(PureFunction.Data, "__pydantic_model__", None)
if pydantic_model:
    pydantic_model.update_forward_refs()
del pydantic_model

class PartialPureFunction(PureFunction):
    """
    A `PartialPureFunction` is a function which, once made partial by binding
    the given arguments, is pure (it has no side-effects).
    The original function may be impure.
    """
    class Data(SerializedData):
        func: Union[str,PureFunction]
        kwargs: dict

        @classmethod
        def encode(cls, purefunc: Union[PartialPureFunction,functools.partial]) -> str:
            if isinstance(purefunc, functools.partial):
                # We need this branch because we register `functools.partial` to PartialPureFunction
                # We want to allow encoding only pure functions, so the wrapped function must be pure
                partialfunc = purefunc
                if isinstance(partialfunc.func, functools.partial):
                    raise NotImplementedError("`PartialPureFunction.Data` does not "
                                              "support nested partial functions at this time")
                elif isinstance(partialfunc.func, PureFunction):
                    # Wrap the partial with PartialPureFunction, and re-apply `encode`
                    return cls.encode(PartialPureFunction(purefunc))
                else:
                    raise TypeError("`PartialPureFunction.Data.encode` can only serialize pure functions")

            # From this point `purefunc` is know to be a PartialPureFunction
            func = purefunc.func
            if isinstance(func, PureFunction):
                return (func, {})  # If `func` wraps a partial, its keywords will be taken care of at a deeper level
            elif not isinstance(func, functools.partial):
                # Serialize the function and store an empty dict of bound arguments
                return (serialize_function(func), {})
                # # Make a partial with empty dict of bound arguments
                # func = functools.partial(func)
            elif isinstance(func.func, functools.partial):
                raise NotImplementedError("`PartialPureFunction.Data` does not "
                                          "support nested partial functions at this time")
            else:
                # At this point we know that `func` is a partial
                if isinstance(func.func, PureFunction):
                    return (func.func, func.keywords)
                else:
                    return (serialize_function(func.func), func.keywords)

        def decode(data: PartialPureFunction.Data) -> PureFunction:
            cls = PartialPureFunction
            modules = [importlib.import_module(m_name) for m_name in cls.modules]
            global_ns = {k:v for m in modules
                             for k,v in m.__dict__.items()}
            if isinstance(data.func, Callable):
                func = data.func
            else:
                func = deserialize_function(data.func, global_ns)
            # if isinstance(func, cls):
            #     raise NotImplementedError(
            #         "Was a partial function saved from function decorated with "
            #         "a PureFunction decorator ? I haven't decided how to deal with this.")
            return PartialPureFunction(functools.partial(func, **data.kwargs))

    # def __init__(self, partial_func):
    #     super().__init__(partial_func)

    # @classmethod
    # def deserialize(cls, value):
    #     if not (isinstance(value, Sequence)
    #             and len(value) > 0 and value[0] == "PartialPureFunction"):
    #         cls.raise_validation_error(value)
    #     assert len(value) == 3, f"Serialized PartialPureFunction must have 3 elements. Received {len(tvalue)}"
    #     assert isinstance(value[1], str), f"Second element of serialized PartialPureFunction must be a string.\nReceived {value[1]} (type: {type(value[1])}"
    #     assert isinstance(value[2], dict), f"Third element of serialized PartialPureFunction must be a dict.\nReceived {value[2]} (type: {type(value[2])})"
    #     func_str = value[1]
    #     bound_values = value[2]
    #     modules = [importlib.import_module(m_name) for m_name in cls.modules]
    #     global_ns = {k:v for m in modules
    #                      for k,v in m.__dict__.items()}
    #     func = deserialize_function(func_str, global_ns)
    #     if isinstance(func, cls):
    #         raise NotImplementedError(
    #             "Was a partial function saved from function decorated with "
    #             "a PureFunction decorator ? I haven't decided how to deal with this.")
    #     return cls(functools.partial(func, **bound_values))


    # @staticmethod
    # def serialize(v):
    #     if isinstance(v, PureFunction):
    #         func = v.func
    #     elif isinstance(v, Callable):
    #         func = v
    #     else:
    #         raise TypeError("`PartialPureFunction.serialize` accepts only "
    #                         "`PureFunction` or Callable arguments. Received "
    #                         f"{type(v)}.")
    #     if not isinstance(func, functools.partial):
    #         # Make a partial with empty dict of bound arguments
    #         func = functools.partial(func)
    #     if isinstance(func.func, functools.partial):
    #         raise NotImplementedError("`PartialPureFunction.serialize` does not "
    #                                   "support nested partial functions at this time")
    #     return ("PartialPureFunction",
    #             serialize_function(func.func),
    #             func.keywords)

PureFunction.Partial = PartialPureFunction
# Allow serializing something like partial(purefunc, a=2)
PartialPureFunction.register(functools.partial)

class CompositePureFunction(PureFunction):
    """
    A lazy operation composed of an operation (+,-,*,/) and one or more terms,
    at least one of which is a PureFunction.
    Non-pure functions are not allowed as arguments.

    Typically obtained after performing operations on PureFunctions:
    >>> f = PureFunction(…)
    >>> g = PureFunction(…)
    >>> h = f + g
    >>> isinstance(h, CompositePureFunction)  # True

    .. important:: Function arithmetic must only be done between functions
       with the same signature. This is NOT checked at present, although it
       may be in the future.
    """
    class Data(SerializedData):
        opname: str
        terms: List[Union[PureFunction, PlainArg]]  # NB: PureFunction first, so it catches str arguments

        def encode(purefunc: CompositePureFunction) -> str:
            assert purefunc.func in operator.__dict__.values(), "CompositePureFunction only supports operations found in the `operator` module."
            return (purefunc.func.__name__, purefunc.terms)
        def decode(data: CompositePureFunction.Data) -> PureFunction:
            cls = CompositePureFunction
            func = getattr(operator, data.opname)
            return cls(func, *data.terms)

    def __new__(cls, func=None, *terms):
        return super().__new__(cls, func)
    def __init__(self, func, *terms):
        if func not in operator.__dict__.values():
            raise TypeError("CompositePureFunctions can only be created with "
                            "functions defined in " "the 'operator' module.")
        for t in terms:
            if isinstance(t, Callable) and not isinstance(t, PureFunction):
                raise TypeError("CompositePureFunction can only compose "
                                "constants and other PureFunctions. Invalid "
                                f"argument: {t}.")
        self.func = func
        self.terms = terms
        if not getattr(self, '__name__', None):
            self.__name__ = "composite_pure_function"

    # TODO? Use overloading (e.g. functools.singledispatch) to avoid conditionals ?
    def __call__(self, *args):
        return self.func(*(t(*args) if isinstance(t, Callable) else t
                           for t in self.terms))

    # @classmethod
    # def deserialize(cls, value):
    #     "Format: ('CompositePureFunction', [op], [terms])"
    #     if not (isinstance(value, Sequence)
    #             and len(value) > 0 and value[0] == "CompositePureFunction"):
    #         cls.raise_validation_error(value)
    #     assert len(value) == 3
    #     assert isinstance(value[1], str)
    #     assert isinstance(value[2], Sequence)
    #     func = getattr(operator, value[1])
    #     terms = []
    #     for t in value[2]:
    #         if (isinstance(t, str)
    #             or isinstance(t, Sequence) and len(t) and isinstance(t[0], str)):
    #             # Nested serializations end up here.
    #             # First cond. catches PureFunction, second cond. its subclasses.
    #             terms.append(PureFunction.deserialize(t))
    #         elif isinstance(t, PlainArg):
    #             # Either Number or Array – str is already accounted for
    #             terms.append(t)
    #         else:
    #             raise TypeError("Attempted to deserialize a CompositePureFunction, "
    #                             "but the following value is neither a PlainArg "
    #                             f"nor a PureFunction: '{value}'.")
    #     return cls(func, *terms)

    # @staticmethod
    # def serialize(v):
    #     if isinstance(v, CompositePureFunction):
    #         assert v.func in operator.__dict__.values()
    #         return ("CompositePureFunction",
    #                 v.func.__name__,
    #                 v.terms)
    #     else:
    #         raise NotImplementedError

#################################################
##               Utilities                     ##
#################################################

import logging
from warnings import warn
import builtins
import operator
import inspect
from itertools import chain

from types import FunctionType
from typing import Callable, Optional

from .config import config
from .utils import UnsafeDeserializationError

logger = logging.getLogger(__name__)

def split_decorators(s):
    s = s.strip()
    decorator_lines = []
    while s[0] == "@":
        line, s = s.split("\n", 1)
        decorator_lines.append(line)
        s = s.lstrip()
        # print(line); print(decorator_lines); print(s)
    return decorator_lines, s


import ast
try:
    import astunparse
except ModuleNotFoundError:
    pass
import textwrap
def remove_comments(s, on_fail='warn'):
    """
    Remove comments and doc strings from Python source code passed as string.
    Based on https://stackoverflow.com/a/56285204

    This function will fail if the string `s` is not valid Python code.

    By default, if an error is raised, the string `s` is
    is returned unchanged and a warning is printed. This follows from the
    idea that `remove_comments` is a 'nice to have' feature, and we would
    rather still save / operator on an unchanged `s` than terminate.

    :param on_fail: Default: 'warn'. Change to 'raise' to raise the error
        instead of simply printing a warning.
    """
    try:
        lines = astunparse.unparse(ast.parse(textwrap.dedent(s))).split('\n')
    except Exception as e:
        if on_fail == 'warn':
            warn(f"{str(e)}\n`remove_comments` encountered the error above. "
                 "The string was returned unmodified.")
            return s
        else:
            raise e
    out_lines = []
    for line in lines:
        if line.lstrip()[:1] not in ("'", '"'):
            out_lines.append(line)
    return '\n'.join(out_lines)

def serialize_function(f) -> str:
    """
    WIP. Encode a function into a string.
    Accepts only definitions of the form::

        def func_name():
            do_something

    or::

        @decorator
        def func_name():
            do_something

    or functions defined in `operator`::

        import operator
        operator.add

    This excludes, e.g. lambdas and dynamic definitions like ``decorator(func_name)``.
    However there can be multiple `@` decorators.

    Upon deserialization, the string is executed in place with :func:`exec`, and the
    user is responsible for ensuring any names referred to within the function
    body are available in the deserializer's scope.

    .. Note:: Instances of `PureFunction` are not supported. Instead use
       `Serializable.json_encoder(pure_function)`, which will take care of
       extracting the function and calling `serialize_function` on it.
    """

    if f in operator.__dict__.values():
        "Special case serializing builtin functions"
        return f"operator.{f.__name__}"
    elif hasattr(f, '__func_src__'):
        return f.__func_src__
    elif isinstance(f, PureFunction):
        raise TypeError("To serialize a `PureFunction`, use `Serializable.json_encoder(purefunc)`.")
        # return f.serialize(f)  # (Partial/Composite)PureFunction takes care of extracting its `func` attribute and calling `serialize_function` on it
    elif isinstance(f, FunctionType):
        s = remove_comments(inspect.getsource(f))
        decorator_lines, s = split_decorators(s)
        if not s.startswith("def "):
            raise ValueError(
                f"Cannot serialize the following function:\n{s}\n"
                "It should be a standard function defined in a file; lambda "
                "expressions are not accepted.")
        return "\n".join(chain(decorator_lines, [s]))
    else:
        raise TypeError(f"Type {type(f)} is not recognized as a "
                        "serializable function.")

def deserialize_function(
    s: str, globals: Optional[dict]=None, locals: Optional[dict]=None):
    """
    WIP. Decode a function from a string.
    Accepts strings of the following forms::

        def func_name(x, y):
            do_something

    or::

        @decorator
        def func_name(x, y):
            do_something
            
    or::
    
        lambda x,y: do_something
        
    or::
    
        x,y -> do_something

    (The last form is equivalent to a lambda function, and is provided as a
    convenience shorthand.)
    Not accepted are dynamic definitions like ``decorator(func_name)``.
    However there can be multiple decorators using the '@' syntax.

    The string is executed in place with :func:`exec`, and the arguments
    `globals` and `locals` can be used to pass defined names.
    The two optional arguments are passed on to `exec`; the deserialized
    function is injected into `locals` if passed, otherwised into `global`.

    .. note:: The `locals` namespace will not be available within the function.
       So while `locals` may be used to define decorators, generally `globals`
       is the namespace to use.

    .. note:: A few namespaces are added automatically to globals; by default,
       these are ``__builtins__``, ``np`` and ``math``. This can
       be changed by modifying the module variable
       `~scityping.config.default_namespace`.

    .. note:: Both `locals` and `globals` will be mutated by the call (in
       particular, the namespaces mentioned above are added to `globals` if not
       already present). If this is not desired, consider making a shallow copy
       of the dicts before passing to `deserialize_function`.
       
    .. note:: While _de_serialization of lambda functions is possible,
       serializing them is not currently supported in general. (This is because
       the output of `inspect.getsource` depends on the context, and we want to
       avoid fragile heuristics.)
       _Re_serialization of a lambda function is possible however.
    """
    msg = ("Cannot decode serialized function. It should be a string as "
           f"returned by inspect.getsource().\nReceived value:\n{s}")
    # First check if this is a builtin; if so, exit early
    if isinstance(s, str) and s.startswith("operator."):
        return getattr(operator, s[9:])
    # Not a builtin: must deserialize string
    if not config.trust_all_inputs:
        raise UnsafeDeserializationError(
            "Deserialization of functions saved as source code requires executing "
            "them with `exec`, and is only attempted if "
            "`scityping.config.trust_all_inputs` is set to `True`.")
    if globals is None and locals is not None:
        # `exec` only takes positional arguments, and this combination is not possible
        raise ValueError("[deserialize]: Passing `locals` argument requires "
                         "also passing `globals`.")
    if isinstance(s, str):
        s_orig = s
        if "def " in s:
            f = _deserialize_def(s, globals, locals)
        elif "lambda " in s or s.count("->") == 1:
            f = _deserialize_lambda(s, globals, locals)
        else:
            raise ValueError(msg)
        # Store the source with the function, so it can be serialized again
        f.__func_src__ = s_orig
        return f
    else:
        raise ValueError(msg)

def _deserialize_def(s, globals, locals):
    decorator_lines, s = split_decorators(s)
    if not s[:4] == "def ":
        raise ValueError(msg)
    fname = s[4:s.index('(')].strip() # Remove 'def', anything after the first '(', and then any extra whitespace
    s = "\n".join(chain(decorator_lines, [s]))
    if globals is None:
        globals = config.default_namespace.copy()
        exec(s, globals)
        f = globals[fname]
    elif locals is None:
        globals = {**config.default_namespace, **globals}
        exec(s, globals)
        f = globals[fname]
    else:
        globals = {**config.default_namespace, **globals}
        exec(s, globals, locals)  # Adds the function to `locals` dict
        f = locals[fname]
    return f
    
def _deserialize_lambda(s, globals, locals):
    # `s` is either of the form `lambda x,…: <expr>` or `x,… -> <expr>`
    s = s.strip()
    if not s.startswith("lambda"):
        assert s.count("->") == 1
        inp, out = s.split("->")
        s = f"lambda {inp}: {out}"
    else:
        assert "->" not in s
    if globals is None:
        globals = config.default_namespace.copy()
    else:
        globals = {**config.default_namespace, **globals}
    return eval(s, globals, locals)
