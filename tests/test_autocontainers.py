"""
In simple cases, we can automatically generate the `Data` container from class annotations.
These are tests for that functionality.
"""
from dataclasses import dataclass
from typing import ClassVar
from pydantic import BaseModel, validator
from scityping import Serializable, Range

def test_autocontainer_plain():
    class Foo(Serializable):
        name: ClassVar[str] = "Foo"
        a: int
        b: str
        c: dict[str, str]
        d: Range
        def __init__(self, a, b, c, d):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
    class Bar(Foo):
        def __init__(self, a, b, c, d):
            super().__init__(a, b, c, d)
            self.b = self.b.upper()
    foo = Foo(3, "beta", {"name": "Nemo"}, range(4, 8))
    dat = foo.deep_reduce(foo)
    foo2 = Foo.validate(dat)
    assert (foo.a == foo2.a
            and foo.b == foo2.b
            and foo.c == foo2.c
            and foo.d == foo2.d)
    bar = Bar(3, "beta", {"name": "Nemo"}, range(4, 8))
    assert bar.b == "BETA"
    bar2 = Serializable.validate(Serializable.deep_reduce(bar))
    assert (bar.a == bar2.a
            and bar.b == bar2.b
            and bar.c == bar2.c
            and bar.d == bar2.d)

def test_autocontainer_dataclass():
    @dataclass
    class Foo(Serializable):
        name: ClassVar[str] = "Foo"
        a: int
        b: str
        c: dict[str, str]
        d: Range
    class Bar(Foo):
        def __init__(self, a, b, c, d):
            super().__init__(a, b, c, d)
            self.b = self.b.upper()
    foo = Foo(3, "beta", {"name": "Nemo"}, range(4, 8))
    dat = foo.deep_reduce(foo)
    foo2 = Foo.validate(dat)
    assert (foo.a == foo2.a
            and foo.b == foo2.b
            and foo.c == foo2.c
            and foo.d == foo2.d)
    bar = Bar(3, "beta", {"name": "Nemo"}, range(4, 8))
    assert bar.b == "BETA"
    foo2 = Foo.validate(dat)
    bar2 = Serializable.validate(Serializable.deep_reduce(bar))
    assert (bar.a == bar2.a
            and bar.b == bar2.b
            and bar.c == bar2.c
            and bar.d == bar2.d)

def test_autocontainer_pydantic():
    class Foo(Serializable, BaseModel):
        name: ClassVar[str] = "Foo"
        a: int
        b: str
        c: dict[str, str]
        d: Range
    class Bar(Foo):
        @validator('b')
        def make_upper(cls, v):
            return v.upper()
    foo = Foo(a=3, b="beta", c={"name": "Nemo"}, d=range(4, 8))
    dat = foo.deep_reduce(foo)
    foo2 = Foo.validate(dat)
    assert (foo.a == foo2.a
            and foo.b == foo2.b
            and foo.c == foo2.c
            and foo.d == foo2.d)
    bar = Bar(a=3, b="beta", c={"name": "Nemo"}, d=range(4, 8))
    assert bar.b == "BETA"
    bar2 = Serializable.validate(Serializable.deep_reduce(bar))
    assert (bar.a == bar2.a
            and bar.b == bar2.b
            and bar.c == bar2.c
            and bar.d == bar2.d)
