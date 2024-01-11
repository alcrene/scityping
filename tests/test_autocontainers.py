"""
In simple cases, we can automatically generate the `Data` container from class annotations.
These are tests for that functionality.
"""
from dataclasses import dataclass
from pydantic import BaseModel
from scityping import Serializable, Range

def test_autocontainer_plain():
    class Foo(Serializable):
        a: int
        b: str
        c: dict[str, str]
        d: Range
        def __init__(self, a, b, c, d):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
    foo = Foo(3, "beta", {"name": "Nemo"}, range(4, 8))
    dat = foo.deep_reduce(foo)
    foo2 = Foo.validate(dat)
    assert (foo.a == foo2.a
            and foo.b == foo2.b
            and foo.c == foo2.c
            and foo.d == foo2.d)

def test_autocontainer_dataclass():
    @dataclass
    class Foo(Serializable):
        a: int
        b: str
        c: dict[str, str]
        d: Range
    foo = Foo(3, "beta", {"name": "Nemo"}, range(4, 8))
    dat = foo.deep_reduce(foo)
    foo2 = Foo.validate(dat)
    assert (foo.a == foo2.a
            and foo.b == foo2.b
            and foo.c == foo2.c
            and foo.d == foo2.d)

def test_autocontainer_pydantic():
    class Foo(Serializable, BaseModel):
        a: int
        b: str
        c: dict[str, str]
        d: Range
    foo = Foo(a=3, b="beta", c={"name": "Nemo"}, d=range(4, 8))
    dat = foo.deep_reduce(foo)
    foo2 = Foo.validate(dat)
    assert (foo.a == foo2.a
            and foo.b == foo2.b
            and foo.c == foo2.c
            and foo.d == foo2.d)
