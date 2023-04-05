import pytest
from typing import Tuple
from scityping.pydantic import BaseModel
from scityping.functions import PureFunction
# from scityping.base_types import Dict

@PureFunction
def f(a):
    return a*2
@PureFunction
def g(a):
    return a/2

def _test_Dict():
    """
    The Dict type is currently WIP because unless we subclass JSONEncoder and
    implement dict serialization ourselves, the serialization of dicts is
    hard-coded.
    """

    class Foo(BaseModel):
        # Key is scalar plain data: also serializable with normal json.dump
        d1: Dict[str,int]
        # Key is tuple: Not serializable with json.dump
        d2: Dict[Tuple[str, int], float]
        # Key is a function: Not serializable with json.dump
        d3: Dict[PureFunction, PureFunction]


    foo = Foo(d1={"a": 3.},
              d2={(1,1): 1},
              d3={f:g})

    # Check that coercion occurred as expected
    assert isinstance(foo.d1["a"], int)
    assert foo.d2 == {("1", 1): 1.}
    assert isinstance(foo.d2[("1", 1)], float)

    # Check serialization round-trip
    foo2 = Foo.parse_raw(foo.json())
    assert foo2.d1 == foo.d1
    assert foo2.d2 == foo.d2
    assert foo2.d3 == foo.d3

    assert isinstance(foo2.d1["a"], int)
    assert foo2.d2 == {("1", 1): 1.}
    assert isinstance(foo2.d2[("1", 1)], float)
