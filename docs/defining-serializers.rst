Defining serializers
====================

Standard: By subclassing the type with the same name
----------------------------------------------------

.. code-block:: python

   class Complex(Serializable, complex):
       @dataclass
       class Data:
           real: float
           imag: float
           def encode(self, z): return z.real, z.imag

By subclassing the type with a different name
---------------------------------------------

- Manually update type registries

.. code-block:: python

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

For types which don’t allow subclassing: register against ABC
--------------------------------------------------------------

- Use ``@ABCSerializable.register``
- Define `decode` so the correct type is returned.

.. code-block:: python

   @ABCSerializable.register
   class Range(Serializable):
       @dataclass
       class Data:
           start: int
           stop: Optional[int]
           step: Optional[int]
           def encode(r): return start, stop, step
           def decode(data): return range(*data)

(NB: If we had called the class ``RangeType`` instead of ``Range``, it would be necessary to also update the registries.)

Registering the same serializer for multiple types
--------------------------------------------------

.. code-block:: python
   
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
