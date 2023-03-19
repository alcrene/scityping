
from dataclasses import dataclass
from scityping import Serializable, Slice, Dataclass, config
from scityping.base import deep_reduce
from scityping.pydantic import BaseModel, dataclass as pydataclass

config.safe_packages.add(__name__)

# Dataclass definitions need to be in global scope in order to be deserializable
@dataclass
class DcStdTypes:
    # i: int
    # s: str
    # f: float
    slc: Slice

@dataclass
class Obj1:
    slc: Slice

def test_dataclass_and_pydantic():
    """
    Tests verifying the interoperability of dataclasses, Pydantic BaseModel and Serializable.
    """
    @dataclass
    class Obj2(Serializable):
        slc: Slice
        @dataclass
        class Data:
            slc: Slice
            def encode(obj):
                return (obj.slc,)

    @pydataclass
    class Obj2b(Serializable):
        slc: Slice
        @dataclass
        class Data:
            slc: Slice
            def encode(obj):
                return (obj.slc,)

    @pydataclass
    class Obj2c(Serializable):
        slc: Slice
        @pydataclass
        class Data:
            slc: Slice
            def encode(obj):
                return (obj.slc,)

    class Obj3(BaseModel):
        slc: Slice

    class Obj4(Serializable):
        slc: Slice
        def __init__(self, slc: slice):
            self.slc = slc
        @dataclass
        class Data:
            slc: Slice
            def encode(obj): return (obj.slc,)

    class Obj4b(Serializable):
        slc: Slice
        def __init__(self, slc: slice):
            self.slc = slc
        @pydataclass
        class Data:
            slc: Slice
            def encode(obj): return (obj.slc,)

    class Foo1(BaseModel):
        obj: Obj1
    class Foo2(BaseModel):
        obj: Obj2
    class Foo2b(BaseModel):
        obj: Obj2b
    class Foo2c(BaseModel):
        obj: Obj2c
    class Foo3(BaseModel):
        obj: Obj3
    class Foo4(BaseModel):
        obj: Obj4
    class Foo4b(BaseModel):
        obj: Obj4b

    foo1 = Foo1(obj=Obj1(slc=slice(3, 8)))
    foo2 = Foo2(obj=Obj2(slc=slice(3, 8)))
    foo2b = Foo2b(obj=Obj2b(slc=slice(3, 8)))
    foo2c = Foo2c(obj=Obj2c(slc=slice(3, 8)))
    foo3 = Foo3(obj=Obj3(slc=slice(3, 8)))
    foo4 = Foo4(obj=Obj4(slc=slice(3, 8)))
    foo4b = Foo4b(obj=Obj4b(slc=slice(3, 8)))

    print("Foo1", Foo1.parse_raw(foo1.json()).obj.__dict__)
    print("Foo2", Foo2.parse_raw(foo2.json()).obj.__dict__)
    print("Foo2b", Foo2b.parse_raw(foo2b.json()).obj.__dict__)
    print("Foo2c", Foo2c.parse_raw(foo2c.json()).obj.__dict__)
    print("Foo3", Foo3.parse_raw(foo3.json()).obj.__dict__)
    print("Foo4", Foo4.parse_raw(foo4.json()).obj.__dict__)
    print("Foo4b", Foo4b.parse_raw(foo4b.json()).obj.__dict__)

    # print(foo2.json())
    # print(Obj2.__validate__)

    # print("Bar1", Bar1.parse_raw(bar1.json()).obj.__dict__)

def test_plain_dataclass():
    """
    Test the special cases supporting serialization of plain dataclasses.
    These only support Serializable types and plain JSON types.
    """
    obj = DcStdTypes(slc=slice(1,2,3))

    Dataclass.reduce(obj) == ("scityping.base_types.Dataclass",
                              ("__main__.DcStdTypes",
                               {"slc": slice(1,2,3)}))
    reduced_data = deep_reduce(obj)
    reduced_data == ("scityping.base_types.Dataclass",
                     ("__main__",
                      {"slc": ("scityping.base_types.Slice",
                               {"start": 1, "stop": 2, "step": 3})}))
    assert Dataclass.deep_reduce(obj) == reduced_data
    objb = Dataclass.validate(reduced_data)
    assert objb.slc == slice(1,2,3)


# if __name__ == "__main__":
    # test_dataclass_and_pydantic()
    # test_plain_dataclass()