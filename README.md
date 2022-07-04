# Scityping: type hints and serializers for scientific types

A collection of type hint specifiers for Python types common in the fields of scientific computing, data science and machine learning. Most types come with a pair of JSON serializer/deserializer functions, which can be used both to archive and transfer data.
In contrast to pickling, the resulting files are both future safe and human readable, which makes them especially suitable to ensure [reproducibility](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html).

Some supported data types (see [below](#defined-types) for a full list):
- `range`
- NumPy arrays[^1]
- NumPy random number generators
- SciPy distributions
- PyTorch tensors
- Values with units (Pint)

[^1]: Large arrays are compressed, but saved alongside a human-readable snippet to allow inspection.

## Compatibility with Pydantic

Types were designed and tested with Pydantic in mind, so they can be used as types in Pydantic `BaseModels`. To support serialization, use
```python
from scityping.pydantic import BaseModel
```
(Don’t worry: this is not a rewrite of `BaseModel`. It provides the same class you get when importing from pydantic, but adds a hook for our extensible type machinery.)

Example usage:

```python
from scityping import Array
from scityping.pydantic import BaseModel

class MyModel(BaseModel):
  data: Array[1, float]  # 1D array of floats

model = MyModel(data=[1, 2, 3])
print(model.data)
# array([1., 2., 3.,], dtype=float)
json_data = model.json()
print(json_data)
#
model2 = MyModel.parse_raw(json_data)
print(model2.data)
# array([1., 2., 3.,], dtype=float)
```

## Usage without Pydantic

Serializers are class methods, so they can also be used by accessing them directly.

```python
import numpy as np
from scityping import Array

data = np.array([1, 1, 2, 3, 5])
json_data = Array.Data.encode(data)
print(json_data)
# 
data2 = Array.validate(json_data)
print(data2)
# array([1, 1, 2, 3, 5], dtype=int)
```

## Origin and motivation

This package is my [fifth](https://github.com/mackelab/mackelab-toolbox/blob/acc2193358f6e2bf9ba7f224d0bbb097b585ce14/mackelab_toolbox/typing.py) [iteration](https://github.com/samuelcolvin/pydantic/issues/951#issuecomment-774297368), [over](https://github.com/mackelab/mackelab-toolbox/tree/0963a2278191537a6cfba6802f9691770567c020/mackelab_toolbox/typing) [three](https://github.com/mackelab/mackelab-toolbox/tree/0963a2278191537a6cfba6802f9691770567c020/mackelab_toolbox/typing "Private link") [years](https://jugit.fz-juelich.de/explainable-ai/statglow/-/tree/develop/statGLOW/smttask_ml/scityping "Private link"), of a simple system for defining Python types that package their own serialization/deserialization routines. The motivating [use case](https://sumatratask.readthedocs.io/) is to support fully self-contained “units of work” for research computations; these are used both to ensure reproducibility by archiving work, and to transfer units of work between processes or machines. (For example between simulation and analysis scripts, or between laptop and server.) This use case imposed the following requirements:

- The serialized format must be portable and suitable for long-term storage (which precludes solutions like pickle or dill). 
- The format should be plain text and, to the extent possible, human-readable. This further increases long-term reliability, since in the worst case, the file can be opened in a text editor and interpreted by a human.
- It must be possible to associate a pair of serialization/deserialization routines to a type.
- These routines must be recognized by [Pydantic](https://pydantic-docs.helpmanual.io/), since this is the serialization library I currently use.
- It should be possible for imported librairies to use custom serializers, without worrying about the order of imports. Note that this is difficult with vanilla Pydantic, because the list of `json_encoders` is set when the class is defined. One can [work around](https://github.com/samuelcolvin/pydantic/issues/951#issuecomment-774297368) this by defining a global dictionary of JSON encoders, but the behaviour is still dependent on the order of imports, and is thus fragile.
- It should be possible to associate custom serializers to already existing types. (E.g. a serializer for `complex`.)
- It should be possible to use generic base classes in a model definition, while subclasses are preserved after deserialization. For example, in the snippet below this is *not* the case:
  ```python
  class Signal:
    ...
  class AMSignal(Signal):
    ...
  class FMSignal(Signal):
    ...

  class MyModel(BaseModel):
    signal: Signal

  sig = AMSignal(...)
  model = MyModel(signal=sig)
  type(model.signal)   # AMSignal
  model2 = MyModel.parse_raw(model.json)
  type(model2.signal)  # Signal -> Serialization round trip does not preserve type.
  ```
  Since it is often the cases that the same computational pipeline can be applied to many different types, the ability to define classes in terms of only generic base types is extremely useful.

The solution provided by this package satisfies all of these requirements.

## An extensible serialization hierarchy

The provided `Serializable` class can be used to make almost any class serializable.
A consistent pattern is used to specify what needs to be serialized, and *scityping* takes care of the rest.

Note that types defined with `Serializable` take precedence over those already supported by Pydantic; this allows to override builtin Pydantic serializers and deserializers for specific types.

Example: Define a serializable `Signal` class.
```python
# mysignal.py
from scityping import Serializable, Array
from scityping.pydantic import dataclass

class Signal(Serializable):

  @dataclass
  class Data:                # The `Data` class stores the minimal
    values: Array[2, float]  # set of information required to
    times : Array[1, int]    # recreate the `Signal` class.
    freq  : float

    @staticmethod
    def encode(signal: Signal):
      return (signal.values, signal.times, signal.freq)

  def __init__(values, times, freq):
    ...

  ## Analysis methods (mean, window, fft, etc.) go here ##
```

Now the class can be used as a type hint in a Pydantic model:
```python
from scityping.pydantic import BaseModel
from typing import List
from mysignal import Signal  # Type defined in the code block above

class Recorder(BaseModel):
  signals: List[Signal]
```

Moreover, *subclasses* of `Signal` can also be used and will be recognized correctly.
```python
class AMSignal(Signal):
  pass

signal = Signal(...)
amsignal = AMSignal(...)

rec = Recorder(signals=[signal, amsignal])
rec2 = Recorder.parse_raw(rec.json())
assert type(rec2.signals[0]) is Signal
assert type(rec2.signals[1]) is AMSignal
```
(With standard Pydantic types, in the example above, values would all be coerced to the `Signal` type after deserialization.)

The default decoder will pass the attributes of the `Data` object to the class’ `__init__` method. For cases where this is unsatisfactory, one can define a custom `decode` method. The example below uses this to return a true `range` object, rather then the `Range` wrapper:
```python
class Range(Serializable, range):
  @dataclass
  class Data:
    start: int
    stop: Optional[int]=None
    step: Optional[int]=None
    def encode(self, r): return r.start, r.stop, r.step
    def decode(self, data: 'Range.Data'): return range(*data)
```
Note that while the `decode` mechanism can help solve corner cases, in most situations the benefits are cosmetic.

Finally, extending an *existing* type with a serializer is just a matter of defining a subclass with the *same name* (case-insensitive). For example, the provided `Complex` type is implemented as follows:
```python
class Complex(Serializable, complex):
  class Data(BaseModel):
    real: float
    imag: float
    def encode(z):
      return z.real, z.imag
```
It can then be used to allow serializing all complex numbers (not just instance of `Complex`):
```python
from scityping.pydantic import BaseModel
class Foo(BaseModel):
  z: Complex

# Serializes even the original data type
Foo(z=3+4j).json()
```
Note that this works even if `Complex` is defined after `Foo`, without any patching of `Foo`’s list of `__json_encoders__`.[^post-type-def]

[^post-type-def]: The corollary of this is that it makes it easier for modules to arbitrarily modify how types are serialized by *already imported* modules. Thus by adding a new serializer, a new package may break a previously working one. Also, while the hooks for extending type suport don't increase the attack surface (imported Python modules are already allowed to inject code wherever they please), they might make it easier for a malicious actor to run arbitrary code when data are serialized – although, if this is a concern for you, you are probably not allowed to install unaudited packages like this one.

## Performance

For types serialized with Pydantic, this adds a single `isinstance` check for each serialization call, which should be negligeable.[^2]

Serialization with `Serializable` is not as aggressively tested with regards to performance and may be a bit slower.

*De*serialization of `Serializable` types may be in some cases be faster: Pydantic fields with `Union` types execute a try-catch for each possible type, keeping the first successful result. Since `Serializable` includes the target type in serialized data, the correct data type is generally attempted first.

In general, within scientific applications these performance considerations should not matter: data is typically serialized/deserialized only at the beginning/end of a computation, which is not where the speed bottlenecks usually are. 

[^2]: Unless *a lot* of builtin types are extended with serializers.

## Defined types

### Base types

- `Serializable`  (mixin class used to define new serializable types)
- `Complex`
- `Range`
- `Slice`
- `Number` (validation only)
- `Integral`  (validation only)
- `Real`  (validation only)

### NumPy types

- `DType`
- `NPValue`
- `Array`
- `NPGenerator`
- `RandomState`

### SciPy types

- `Distribution`
  + `UniDistribution`
  + `MvNormalDistribution`

### PyTorch types

- `TorchTensor`
- `TorchModule` (partially; validation + separate serialization functions)
- `TorchGenerator`