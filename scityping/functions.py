from __future__ import annotations

import sys
import importlib
import inspect
import operator
import functools
import re
from collections.abc import Callable as Callable_

from typing import ClassVar, Union, Type, Callable, List, Tuple, _Final
from types import new_class

import numpy  # TODO: Make numpy import optional

from .base import Serializable
from .base_types import SerializedData, Number
from .numpy import Array  # TODO: Make numpy import optional

PlainArg = Union[Number, str, Array]

__all__ = ["PlainArg", "PureFunction", "PartialPureFunction", "CompositePureFunction"]

# Functions from these modules are not serialized by storing their source,
# but simply the name is saved. There are a few reasons to do this:
# - C functions, like the builtins, cannot be serialized by inspecting their source.
# - NumPy ufuncs similarly cannot be serialized by inspection
# - NumPy array functions technically could, but because of the overloading the
#   dependencies are quite complicated.
# - Serializing just the name is much more compact, and arguably more reliably
#   for mature functions like those in core Python or NumPy
pure_function_set = {fn: module.__name__
                     for module in [operator, numpy]
                     for fn in module.__dict__.values()
                     if isinstance(fn, Callable_)
                     }


# TODO: Instead of "trust_all_inputs", use a whitelist of safe module (as in scityping.scipy.Distribution)
# TODO: Serializing partial(PureFunction(g)) should be the same as PartialPureFunction(partial(g)), for consistent hashes
# FIXME!: Specifying types to a PureFunction multiple times causes the following error:
#       The Serializable TypeRegistry already contained an entry for key
#       scityping.functions.purefunctionmeta.__getitem__.<locals>.update_namespace.<locals>.data,
#       which has been replaced.

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
            __args__ = (*callableT['inT'], callableT['outT'])
                # Not sure that merging inT and outT is the best, but 
                # typing.Callable does it like this
        else:
            baseT = Callable
            argstr = ""
            __args__ = ()
        # Treat the module names, if present
        if modules:
            if argstr:
                argstr += ", "
            argstr += ", ".join(modules)
        # Check if this PureFunction has already been created, and if not, do so
        key = (cls, baseT, tuple(modules))
        # We set Serializable's subtype registry key ourselves, otherwise the 
        # automatically determined one looks like scityping.functions.purefunctionmeta.__getitem__.<locals>.update_namespace.<locals>.data
        # which is not informative and causes name conflicts.
        if callableT['inT']:
            serialization_key = f"{cls.__qualname__}[[{','.join(T.__qualname__ for T in callableT['inT'])}], {callableT['outT'].__qualname__}]"
        else:
            serialization_key = cls.__qualname__  # Equivalent to letting utils.get_type_key automatically determine the key
        if key not in cls._instantiated_types:
            def update_namespace(namespace):
                # All subclasses of Serializable need their own nested Data class
                class Data(cls.Data):
                    __serialization_key__ = serialization_key + ".Data"
                namespace["modules"] = cls.modules + modules
                namespace["Data"] = Data
                namespace["__args__"] = __args__
                namespace["__serialization_key__"] = serialization_key
                namespace["__module__"] = cls.__module__
            PureFunctionSubtype = new_class(
                f'{cls.__name__}[{argstr}]', (cls,),
                exec_body=update_namespace)
            cls._instantiated_types[key] = PureFunctionSubtype
            # To also allow an instance of an untyped PureFunction as a value for
            # a typed field, we need to add PureFunction to the subclass' _registry
            # (See docs/implementation_logic.md::Validation of functions)
            PureFunctionSubtype.register(cls)
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
       
       To provide additional scope, one or more modules name can be added to the
       list `PureFunction.modules`. These modules are imported deserialization
       but are not serialized with the function.
       Adding modules this way is considered an experimental feature.


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

    .. Hint:: `PureFunction` instances can be pickled, using the same serializer
       as `Serializable.reduce`. This can make them more portable:
       whereas a pickled python function requires that the original function &
       module still exist at the same location, a pickled `PureFunction`
       deserializes from the source code in the serialized data.
    """
    __args__: ClassVar[Tuple[Type,...]] = ()
    modules: List[str] = []  # Use this to list modules that should be imported into
                             # the global namespace before deserializing the function
    # Instance variable
    func: Callable

    class Data(SerializedData):
        func: Union[Callable]
        def __post_init__(self):
            cls = PureFunction  # TODO? Get the container of type(self) ?
            if not isinstance(self.func, Callable_):
                modules = [importlib.import_module(m_name) for m_name in cls.modules]
                global_ns = {k:v for m in modules
                                 for k,v in m.__dict__.items()}
                # Since decorators are serialized with the function, we should at
                # least make the decorators in this module available.
                local_ns = {'PureFunction': PureFunction,
                            'PartialPureFunction': PartialPureFunction,
                            'CompositePureFunction': CompositePureFunction}
                self.func = deserialize_function(self.func, global_ns, local_ns)
            super().__post_init__()

        def encode(purefunc: PureFunction) -> str:
            return purefunc.func,

    def __new__(cls, func=None):
        # func=None allowed to not break __reduce__ (due to metaclass)
        # – inside a __reduce__, it's fine because __reduce__ will fill __dict__ after creating the empty object
        if isinstance(func, functools.partial) and not issubclass(cls, PartialPureFunction):
            # Redirect to PartialPureFunction constructor
            # FIXME: What if we get here from CompositePureFunction.__new__
            try:
                Partial = cls.__dict__['Partial']
            except KeyError:
                raise TypeError(f"{func} is a partial function but '{cls}' "
                                "does not define a partial variant.")
            else:
                return Partial(func)
        return super().__new__(cls)
    def __init__(self, func):
        if hasattr(self, 'func'):
            # This is our second pass through __init__, probably b/c of __new__redirect
            # assert hasattr(self, '__signature__')
            return
        if isinstance(func, str):
            func = deserialize_function(func)
        self.func = func
        # Copy attributes like __name__, __module__, ...
        functools.update_wrapper(self, func)
        try:
            self.__signature__ = inspect.signature(func)
        except ValueError:  # Some functions, like numpy.sin, don’t support `signature`
            pass
    
    @classmethod
    def validate(cls, value, field=None):  # `field` not currently used: only there for consistency
        # Extra check: Make sure the wrapped function is pure
        # (validation = deserialization, so the function should already be marked pure)
        if isinstance(value, Callable_):
            if not is_pure_function(value):
                raise TypeError(f"Cannot validate {value} as a PureFunction, "
                                "because callables are assumed non-pure by default. "
                                "To indicate that a function is pure, use the "
                                "`scityping.functions.PureFunction` decorator.")
        # Replace / Augment branch 2: If we use a PureFunction to validate a
        # field with type PureFunction[[int], None], we don’t want this to fail
        # outright
        # !! WIP !! With more work, we could be both more stringent (allow less
        # errors) and less stringent (allow things which do fit the pattern.)
            if not matching_signature(value, cls):
                raise TypeError(f"The signature of {value} does not match "
                                f"that required by the type {cls}.\n"
                                "Note that signature checking is still primitive: "
                                "for example, it does not match `int` to "
                                "`Union[int,float]`to bypass the check, use "
                                "`PureFunction` without specifying a signature")
        # Additional shorthand specification format like: "x,y -> do_something"
        elif isinstance(value, str):                           # Instead of checking that string looks valid, let `deserialize_function` fail if necessary
            value = PureFunction(deserialize_function(value))  # This should produce more useful error messages.
        # Continue with the normal validation
        return super().validate(value, field)
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return f"{type(self).__name__}({self.func})"

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

    # TODO: Add arg so PureFunction subtype can be specified in error message
    @classmethod
    def raise_validation_error(cls, value):
        raise TypeError("PureFunction can be instantiated from either "
                        "a callable, "
                        "a Sequence `([PureFunction subtype name], func, bound_values)`, "
                        "or a string. "
                        f"Received {value} (type: {type(value)}).")

# # If pydantic is available, it is used for the Data classes, and we need to resolve the circular dependency
# pydantic_model = getattr(PureFunction.Data, "__pydantic_model__", None)
# if pydantic_model:
#     pydantic_model.update_forward_refs()
# del pydantic_model

class PartialPureFunction(PureFunction):
    """
    A `PartialPureFunction` is a function which, once made partial by binding
    the given arguments, is pure (it has no side-effects).
    The original function may be impure.
    """
    class Data(SerializedData):
        func: Union[Callable]
        args: tuple
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
                return (func, (), {})
                # # Make a partial with empty dict of bound arguments
                # func = functools.partial(func)
            elif isinstance(func.func, functools.partial):
                raise NotImplementedError("`PartialPureFunction.Data` does not "
                                          "support nested partial functions at this time")
            else:
                # At this point we know that `func` is a partial
                if isinstance(func.func, PureFunction):
                    return (func.func, func.args, func.keywords)
                else:
                    return (func.func, func.args, func.keywords)

        def decode(data: PartialPureFunction.Data) -> PureFunction:
            return PartialPureFunction(functools.partial(data.func, *data.args, **data.kwargs))

    def __str__(self):
        func, args, kwargs = self.Data.encode(self)
        return f"{type(self).__name__}({func}, {args}, {kwargs})"

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
    >>> isinstance(h, CompositePureFunction)
    True

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

    def __str__(self):
        return f"{type(self).__name__}({self.func}, {self.terms})"

    # TODO? Use overloading (e.g. functools.singledispatch) to avoid conditionals ?
    def __call__(self, *args):
        return self.func(*(t(*args) if isinstance(t, Callable) else t
                           for t in self.terms))


#################################################
##               Utilities                     ##
#################################################

import logging
logger = logging.getLogger(__name__)
from warnings import warn
import builtins
import operator
import inspect
from itertools import chain

from types import FunctionType
from typing import Callable, Optional, Any

import ast
if not hasattr(ast, "unparse"): # unparse is available only for >3.9, but the astunparse package backports to earlier versions
    try:
        import astunparse
    except ModuleNotFoundError:
        logger.warning("You may want to install `astunparse` for more reliable "
                       "function serialization, or upgrade to Python ≥3.9.")
    else:
        ast.unparse = astunparse.unparse
import textwrap

from .config import config
from .utils import UnsafeDeserializationError

def is_pure_function(f: Union[Callable, Any]) -> bool:
    """
    Return True if `f` is (very probably)[#]_ a pure function.
    Whether a function is pure is very difficult to determine automatically,
    and we rely on users to identify which functions are pure. Specifically,
    this returns `True` if one of following conditions is met:

    - `f` is not callable (trivial case)
    - `f` was decorated with `PureFunction`.
    - `f` is defined in one of the following modules, which are know to
      define pure functions:
        + `operator`
        + `numpy` (global namespace only)
    - `f` is a partial function, and original function is pure.
      This is checked by calling `is_pure_function` recursively.
      (corner case: we also check bound arguments, in case on of them is a function)

    In all other situations, the returned value will be False.

    .. [#] We say “very probably” because it is possible, for example, to
       decorate a non-pure function with `PureFunction`, or to monkey patch a
       non-pure function into `operator`. In practice, as long as the user does
       not engage in such self-destructive behaviour, and is careful to only
       mark as pure functions which really are pure, this function is reliable.
    """
    return (not isinstance(f, Callable_)
            or isinstance(f, PureFunction)
            or f in pure_function_set
            or (isinstance(f, functools.partial)
                and is_pure_function(f.func)
                and all(is_pure_function(a) for a in f.args)
                and all(is_pure_function(kw) for kw in f.keywords))
            )

def matching_signature(f: Callable, CallableT: Type):
    """
    Return `True` if the signature of `f` matches the type of `CallableT`.
    `CallableT` would normally be created as either `Callable[[int, int] bool]`
    or `PureFunction[[int, int] bool]`.

    .. CAUTION:: This function currently only supports exact type matches in
       the signature. So if the type is `Callable[[Union[int,float]], bool]`
       and the given function is `def f(a:int) -> bool`, it will fail
       becaues `int == Union[int,float]` returns `False`.
       However, if the function omits type hints for certain arguments, then
       those arguments always match.
    """
    if not getattr(CallableT, "__args__", None):
        return True  # If no types are given, everything matches

    inT = CallableT.__args__[:-1]
    outT = CallableT.__args__[-1]
    sig = inspect.signature(f)
    pos_only_params = {nm: p for nm, p in sig.parameters.items()
                       if p.kind == p.POSITIONAL_ONLY}
    pos_or_kw_params = {nm: p for nm, p in sig.parameters.items()
                        if p.kind == p.POSITIONAL_OR_KEYWORD}
    kw_only_params = {nm: p for nm, p in sig.parameters.items()
                      if p.kind == p.KEYWORD_ONLY}
    req_params = {nm: p for nm, p in sig.parameters.items()
                  #if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
                  if p.default is inspect._empty}
    # TODO? Do something with VAR_POSITIONAL argument ?

    # Check that the number of args match
    if (len(inT) < len(req_params)
          or len(inT) > len(pos_only_params) + len(pos_or_kw_params) + len(kw_only_params)):
        return False

    # If available, check that the output type matches
    if sig.return_annotation is not inspect._empty:
        if sig.return_annotation != outT:
            return False

    # Check that the positional only arguments types match (fixed order)
    for p, T in zip(pos_only_params.values(), inT):
        if p.annotation is inspect._empty:
            continue
        elif p.annotation != T:
            return False

    # Check that the other arguments match.
    # Since they can be passed by keyword, the order does not matter
    # First we check that all required args in the signature match a type in CallableT
    nonpos_inT = list(inT[len(pos_only_params):])
    req_Ts = [p.annotation for p in sig.parameters.values()
              if (p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
                  and p.default is inspect._empty
                  and p.annotation is not inspect._empty)]
    opt_Ts = [p.annotation for p in sig.parameters.values()
              if (p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
                  and p.default is not inspect._empty
                  and p.annotation is not inspect._empty)]
    no_ann = [p for p in sig.parameters.values()
              if (p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}
                  and p.annotation is inspect._empty)]
    for sigT in req_Ts:
        try:
            nonpos_inT.remove(sigT)
        except ValueError:
            # The function has a required arg which doesn’t match a type in CallableT
            return False

    # Any remaining types in CallableT must match one in the signature
    # Exception: if an argument has no type annotation, it can match any arg in CallableT
    missed_matches = 0
    for annT in nonpos_inT:
        try:
            opt_Ts.remove(annT)
        except ValueError:
            # CallableT has an arg which doesn’t match one in signature, even an optional one
            missed_matches += 1
    # Number of missed matches must not exceed number of non-annotated arguments in signature
    if missed_matches > len(no_ann):
        return False

    # Passed all the tests => return True
    return True

def split_decorators(s):
    s = s.strip()
    decorator_lines = []
    while s[0] == "@":
        line, s = s.split("\n", 1)
        decorator_lines.append(line)
        s = s.lstrip()
        # print(line); print(decorator_lines); print(s)
    return decorator_lines, s

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
    # # The bit below is based on a more recent answer to the same question: https://stackoverflow.com/a/76593305
    # # It might be more robust, but since I don’t know ast (and even less astor), I would need to test more thoroughly
    # try:
    #     parsed = ast.parse(s)
    # except Exception as e:
    #     if on_fail == 'warn':
    #         warn(f"{str(e)}\n`remove_comments` encountered the error above. "
    #              "The string was returned unmodified.")
    #         return s
    #     else:
    #         raise e
    # for node in ast.walk(parsed):
    #     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
    #         # set value to empty string
    #         node.value = ast.Constant(value='') 
    # formatted_code = ast.unparse(parsed)  
    # pattern = r'^.*"""""".*$' # remove empty """"""
    # formatted_code = re.sub(pattern, '', formatted_code, flags=re.MULTILINE) 
    # return formatted_code

    try:
        src = ast.unparse(ast.parse(textwrap.dedent(s)))
    except Exception as e:
        if on_fail == 'warn':
            warn(f"{str(e)}\n`remove_comments` encountered the error above. "
                 "The string was returned unmodified.")
            return s
        else:
            raise e
    # Although this looks different than the quoted answer, the logic is the
    # same: remove lines which start with '"'
    # However we also identify multiline docstrings, something the original code missed
    # Remove multiline strings
    src = re.sub(r'^\s*"""[\s\S]*?"""[ \t]*\n?', "", src, flags=re.MULTILINE)
    src = re.sub(r"^\s*'''[\s\S]*?'''[ \t]*\n?", "", src, flags=re.MULTILINE)
    # Remove single line strings  (must do this after multiline strings !)
    src = re.sub(r'^\s*".*\n', "", src, flags=re.MULTILINE)
    src = re.sub(r"^\s*'.*\n", "", src, flags=re.MULTILINE)
    return src

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
       `Serializable.reduce(pure_function)`, which will take care of
       extracting the function and calling `serialize_function` on it.
    """

    if f in pure_function_set:
        # Special case serializing of standard pure functions
        return f"{pure_function_set[f]}.{f.__name__}"  # Only support __name__: In `deserialize_function`, we split on the last period to get the function name
    # if f in operator.__dict__.values():
    #     "Special case serializing builtin functions"
    #     return f"operator.{f.__name__}"
    elif hasattr(f, '__func_src__'):
        return f.__func_src__
    elif isinstance(f, PureFunction):
        raise TypeError("To serialize a `PureFunction`, use `Serializable.reduce(purefunc)`.")
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
    # First check if this is a special-cased function; if so, exit early
    if isinstance(s, str) and " " not in s:  # Any function def should have white space, and no identifier can have white space
        modname, fname = s.rsplit(".", 1)       
        if modname not in sys.modules:
            raise RuntimeError(f"Cannot deserialize function '{s}': "
                               f"module {modname} is not imported.")
        return getattr(sys.modules[modname], fname)
    # if isinstance(s, str) and s.startswith("operator."):
    #     return getattr(operator, s[9:])
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
        msg = ("Cannot decode serialized function. It should be a string as "
               f"returned by inspect.getsource().\nReceived value:\n{s}")
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
