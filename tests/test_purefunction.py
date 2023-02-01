import pytest
from functools import partial
from pydantic import ValidationError
import scityping
from scityping.base import Serializable
from scityping.utils import UnsafeDeserializationError
from scityping.pydantic import BaseModel
from scityping.purefunction import PureFunction, PartialPureFunction, CompositePureFunction

@PureFunction
def pure_f(x, n=2):
    return x/n if x%n else x**n

def f(x, n=2):
    return x/n if x%n else x**n

class Foo(BaseModel):
    f: PureFunction

def test_purefunction():

    # Test reduction/reconstruction with Data object
    # Test that `config.trust_all_inputs` option works as expected
    data = Serializable.json_encoder(pure_f)
    with pytest.raises(UnsafeDeserializationError):
        Serializable.validate(data)
    scityping.config.trust_all_inputs = True
    f2 = Serializable.validate(data)
    assert all(pure_f(x) == f2(x) for x in [3, 5, 8, 100, 101])

    # Nested PureFunctions get collapsed, so naturally they serialize fine
    fpp = PureFunction(PureFunction(pure_f))
    datapp = Serializable.json_encoder(fpp)
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
        Serializable.json_encoder(f)
    with pytest.raises(ValidationError):
        Foo(f=f)

def test_partialpurefunction():
    # If `partial` wraps a non-pure function, it cannot be serialized
    with pytest.raises(TypeError):
        Serializable.json_encoder(partial(f, n=3))

    # If `partial` wraps a *pure* function, it *can* be serialized
    g = partial(pure_f, n=3)
    data = Serializable.json_encoder(g)
    g2 = Serializable.validate(data)
    assert all(g(x) == g2(x) for x in [3, 5, 8, 100, 101])

    # We can also directly declare a PartialPureFunction
    h = PartialPureFunction(partial(f, n=4))
    data = Serializable.json_encoder(h)
    h2 = Serializable.validate(data)
    assert all(h(x) == h2(x) for x in [3, 5, 8, 100, 101])

    # Test full round-trip serialization to JSON
    foo = Foo(f=g)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

    foo = Foo(f=h)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])

def test_compositepurefunction():
    g = partial(pure_f, n=3)
    h = PartialPureFunction(partial(f, n=4))
    # Test arithmetic
    with pytest.raises(TypeError):  # At least for now, both arguments need to be PureFunctions
        w = pure_f + g      # (here g is just partial (although it wraps a PureFunction))
    w = pure_f + h          # (here h is a proper PureFunction)
    assert isinstance(w, CompositePureFunction)
    assert all(w(x) == f(x)+h(x) for x in [3, 5, 8, 100, 101])

    # Test reduction/reconstruction with Data object
    data = Serializable.json_encoder(w)
    w2 = Serializable.validate(data)
    assert all(w(x) == w2(x) for x in [3, 5, 8, 100, 101])

    # Test full round-trip serialization to JSON
    foo = Foo(f=w)
    foo2 = Foo.parse_raw(foo.json())
    assert all(foo.f(x) == foo2.f(x) for x in [3, 5, 8, 100, 101])
