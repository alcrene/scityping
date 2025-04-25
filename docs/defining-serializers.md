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
       def encode(z): return z.real, z.imag
```

````{admonition} *Scityping* shorthands
:class: hint

You may notice that the `encode` method above has no `self` argument.
This is because *scityping* internally translates it to:

```python
class Complex(Serializable, complex):
   @dataclass
   class Data:
       real: float
       imag: float
       @classmethod
       def encode(datacls, z): return datacls(z.real, z.imag)
```

Sometimes using the more explicit form with `@classmethod` is useful.
On the other hand, when the data types are very simple, reducing boilerplate
can improve readability.

Note also that it would make no sense to pass `self` to `encode`, since the `Data`
object is not yet initialized. If it helps, you can think of `encode` as a
combination of `__new__` and `__init__` methods.

````

By subclassing the type with a different name
---------------------------------------------

- Manually update type registries

```python

class NPGenerator(Serializable, np.random.Generator):
   @dataclass
   class Data:
       state: dict
       def encode(rng): return rng.bit_generator.state
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
