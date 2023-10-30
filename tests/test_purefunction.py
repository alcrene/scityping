import pytest
from functools import partial
import operator
import numpy as np
from pydantic import ValidationError
import scityping
from scityping.base import Serializable
from scityping.utils import UnsafeDeserializationError
from scityping.pydantic import BaseModel
from scityping.functions import PureFunction, PartialPureFunction, CompositePureFunction

@PureFunction
def pure_f(x, n=2):
    """This is

    a multiline
    docstring
    """
    return x/n if x%n else x**n

def f(x, n=2):
    return x/n if x%n else x**n

class Foo(BaseModel):
    f: PureFunction

def test_purefunction():

    # Test reduction/reconstruction with Data object
    # Test that `config.trust_all_inputs` option works as expected
    data = Serializable.deep_reduce(pure_f)
    with pytest.raises(UnsafeDeserializationError):
        Serializable.validate(data)
    scityping.config.trust_all_inputs = True
    f2 = Serializable.validate(data)
    assert all(pure_f(x) == f2(x) for x in [3, 5, 8, 100, 101])

    # Nested PureFunctions get collapsed, so naturally they serialize fine
    fpp = PureFunction(PureFunction(pure_f))
    datapp = Serializable.deep_reduce(fpp)
    assert datapp == data
    fpp2 = Serializable.validate(datapp)
    assert all(fpp(x) == fpp2(x) for x in [3, 5, 8, 100, 101])

    # Test full round-trip serialization to JSON
    foo = Foo(f=pure_f)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

    # We want to force users to indicate that functions are pure;
    # therefore plain undecorated functions don't serialize.
    with pytest.raises(TypeError):
        Serializable.reduce(f)
    with pytest.raises(ValidationError):
        Foo(f=f)

    # We also support initializing from a string with the format
    f3 = PureFunction.validate("x -> x/(1+x)")
    foo = Foo(f=f3)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

def test_partialpurefunction():
    scityping.config.trust_all_inputs = True
    
    # If `partial` wraps a non-pure function, it cannot be serialized
    with pytest.raises(TypeError):
        Serializable.reduce(partial(f, n=3))

    # If `partial` wraps a *pure* function, it *can* be serialized
    # (Serializes as `PartialPure(f, kwds)`
    g = partial(pure_f, n=3)
    data = Serializable.reduce(g)
    g2 = Serializable.validate(data)
    assert all(g(x) == g2(x) for x in [3, 5, 8, 100, 101])

    # We can also directly declare a PartialPureFunction
    # (Serializes as `PartialPure(Pure, kwds)`. Note that this is different but equivalent to the case above.)
    h = PartialPureFunction(partial(f, n=4))
    data = Serializable.reduce(h)
    h2 = Serializable.validate(data)
    assert all(h(x) == h2(x) for x in [3, 5, 8, 100, 101])

    # Test full round-trip serialization to JSON
    foo = Foo(f=g)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

    foo = Foo(f=h)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

    # We also support initializing from a string with the format
    f3 = PureFunction.validate("x,y -> x/(1+x)")
    fa = partial(f3, 2)
    fb = partial(f3, y=2)

    foo = Foo(f=fa)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])
    foo = Foo(f=fb)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

def test_compositepurefunction():
    scityping.config.trust_all_inputs = True
    
    g = partial(pure_f, n=3)
    h = PartialPureFunction(partial(f, n=4))
    # Test arithmetic
    with pytest.raises(TypeError):  # At least for now, both arguments need to be PureFunctions
        w = pure_f + g      # (here g is just partial (although it wraps a PureFunction))
    w = pure_f + h          # (here h is a proper PureFunction)
    assert isinstance(w, CompositePureFunction)
    assert all(w(x) == f(x)+h(x) for x in [3, 5, 8, 100, 101])

    # Test reduction/reconstruction with Data object
    data = Serializable.reduce(w)
    w2 = Serializable.validate(data)
    assert all(w(x) == w2(x) for x in [3, 5, 8, 100, 101])

    # Test full round-trip serialization to JSON
    foo = Foo(f=w)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

def test_special_functions():
    """
    Test built-in serialization support for common functions like
    NumPy ufuncs and array functions.
    """
    # ufunc: np.sin, np.add
    # array function: np.max, np.clip
    for fn in [operator.abs,
               np.sin,
               np.max,
               partial(np.add, 3),
               partial(np.clip, a_min=-1, a_max=2),
               partial(operator.pow, 2),
               "x -> np.tanh(x)"
               ]:
        purefn = PureFunction(fn)
        test_vals = [-10, -0.3, 0, 1, 5]
        # Test reduction/reconstruction with Data object
        data = Serializable.reduce(purefn)
        purefn2 = Serializable.validate(data)
        assert all(purefn(x) == purefn2(x) for x in test_vals)
        # Test full round-trip serialization to JSON
        foo = Foo(f=purefn)
        foo2 = Foo.parse_raw(foo.json())
        assert all(purefn(x) == foo.f(x) == foo2.f(x) for x in test_vals)

