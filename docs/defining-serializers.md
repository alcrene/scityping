Defining serializers for preexisting types
==========================================

A key feature of *scityping* is that it allows not only to make your own types serializable (by subclassing {py:class}`scityping.Serializable`), but also to associate serializers to preexisting types. This is especially useful to add support for types defined in external libraries, over which we do not have control.

Below we list different approaches for achieving this.
The simplest approach is to subclass the preexisting type, and to reuse its name in for the subclass. Other approaches are more flexible but slightly more verbose.

Standard: By subclassing the type with the same name
----------------------------------------------------

```python
class Complex(Serializable, complex):
   @dataclass
   class Data:
       real: float
       imag: float
       def encode(self, z): return z.real, z.imag
```

By subclassing the type with a different name
---------------------------------------------

- Manually update type registries

```python

class NPGenerator(Serializable, np.random.Generator):
   @dataclass
   class Data:
       state: dict
       def encode(self, rng): return rng.bit_generator.state
       def decode(data):
           bg = getattr(np.random, data.state["bit_generator"])()
           bg.state = data.state
           return np.random.Generator(bg)

NPGenerator.register(np.random.Generator)
```

For types which don’t allow subclassing: register against ABC
--------------------------------------------------------------

- Use `@ABCSerializable.register`
- Define `decode` so the correct type is returned.

```{margin}
(If we had called the class `RangeType` instead of `Range`, it would be necessary to also update the registries.)
```
```python
@ABCSerializable.register
class Range(Serializable):
   @dataclass
   class Data:
       start: int
       stop: Optional[int]
       step: Optional[int]
       def encode(r): return start, stop, step
       def decode(data): return range(*data)
```


Registering the same serializer for multiple types
--------------------------------------------------

```python
import numpy as np

class NPType(Serializable, np.generic):
   @dataclass
   class Data:
       nptype: str
       value: Union[float, int, str]
   def encode(val): return type(val).__name__, val.item()
   def decode(data): return getattr(np, data.nptype)(data.value)

for type_name in ('int8', 'int16', 'int32', 'int64',
                 'float16', 'float32', 'float64', 'float128',
                 'complex64', 'complex128', 'complex256',
                 'str_'):
   NPType.register(getattr(np, type_name))
```
