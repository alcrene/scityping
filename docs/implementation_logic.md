# Implementation logic

```{note}
The documentation on this page is meant for developers, and for people who want to know exactly how validation is implemented. It should not be required for standard use.
```

## Validation logic

The goal is that when one calls `TargetType.validate(obj)`, either the resulting object an instance of `TargetType` (possibly a subclass), or an error is raised. This should work whether `obj` is already of type `TargetType`, of some different type that can be cast to `TargetType`, or some serialized data compatible with `TargetType`. Thus the validation logic must check for all this possibilities, and implements different validation procedures for each. This is implemented in {py:meth}`scityping.base.Serializable.validate`; a schematic is shown below:

```{image} ./validation_logic.svg
:alt: Diagram of the validation logic
:class: full-width
:width: 100%
:align: left
```

This logic relies on two registries:
- A registry of *base types*, composed of `{S: (Q1,Q2,…)}` pairs mapping serializable types to their subclasses.  
  This is labelled $\mathcal{B}$ in the diagram, and `ABCSerializable._base_types` in the code.
  We can think of this as maintaining the parent class → child class relation, whereas standard Python typically only maintains the child → parent relation.
- A registry of *serializable subtypes*, composed of `{A: S}` pairs mapping an arbitrary type to a *serializable* one.
  For example, the pair `{numpy.ndarray: Array}` indicates that the type `Array` provides a serializer for `numpy.ndarray`.
  This registry is labeled $\mathcal{R}_t$ in the diagram (where $t$ indicates the subclass of `Serializable` to which the registry is attached), and `<Serializable subclass>._registry` in the code.

  For each serializable type `S`, {py:func}`scityping.utils.get_type_key` returns a unique key string; this key serialized along with the data, and allows the validation logic to determine exactly which deserializer to use.  
  Each serializable type maintains its own subtype registry, so that entries in the registry for `Array` only correspond to subclasses of `Array`, or at least types which are Liskov-substitutable.  

### Validation of functions

In analogy with `typing`'s `Callable`, {py:class}`scityping.functions.PureFunction` can be used either on its own or by specifying the argument and output types:

```python
class MyClass(Serializable):
    class Data(SerializedData)
        f1: PureFunction
        f2: PureFunction[[int, float], bool]
```

The rules for validation are as one would expect:
- `PureFunction` accepts any pure function
- `PureFunction[[int, float], bool]` accepts any pure function whose signature matches `(int, float) -> bool`.
  Importantly, an instance of `PureFunction` is still accepted, if it's signature matches.

This last point complicates the implementation, because it means that we have two-way equivalence between types (`PureFunction[[int, float], bool]` can accept a `PureFunction`, and vice versa). This is examplified in the diagram below (`A -> B` means “B annotation accepts A value”), and it means that the “parent accepts child” pattern we usually assume for compatibility is insufficient.

```{mermaid}
flowchart LR
    subgraph abstract base type
        PF[PureFunction]
    end
    subgraph "typed functions (examples)"
        PFif["PureFunction[[int,float],bool]"]
        PFii["PureFunction[[int,int],bool]"]
    end
    subgraph "abstract types (implementation details)"
        PPF[PartialPureFunction]
        CPF[CompositePureFunction]
    end
    PF <--> PFif
    PF <--> PFii
    PPF --> PF
    CPF --> PF
    PPF --> PFif
    PPF --> PFii
    CPF --> PFif
    CPF --> PFii
```

The solution however is quite simple: when we create a typed subclass like `PureFunction[[int,float],bool]`, we add `PureFunction` to its subtype dictionary ($\mathcal{R}_t$, `_registry`). The wrapper for `PureFunction.validate` then ensures that only functions matching the signature are accepted.

## Serialization logic

An important goal of *scityping* is to allow writing code that will serialize variables without knowing in advance what the type of those variables are. This is implemented via a dispatch mechanism in the encoding method of the base class, {py:meth}`Serializable.reduce`. Given subclass `S` ⩽ `Serializable`, then `S.reduce(obj)` will do the following:
- Look through `S._registry` for an entry `(T, S')` for which `T` matches `type(obj)`, or any of its parent types. Precedence is given more specific types. 
- Serialize `obj` into `data`, using `S'.Data.encode`.
- Return the tuple `(get_type_key(T), data)`.

Crucially, type name matching allows for fuzzy matches, so the key `"scityping.numpy.Generator"` will also match the type `numpy.random.Generator`; i.e. some tokens separated by `"."` may be omitted. Tokens are matched case-insensitive, must preserve order, must match on the rightmost token,[^rightmost] and must result in a unique match. Some examples

| Subtype Registry                               | `obj` type             | Matches           |
|:-----------------------------------------------|:-----------------------|:------------------|
| {"Complex": \*}                                | complex                | "Complex"         |
| {"Juniper": \*}                                | complex                | raises `KeyError` |
| {"Generator": \*}                              | numpy.random.Generator | "Generator"       |
| {"gENeRatOR": \*}                              | numpy.random.Generator | "gENeRatOR"       |
| {"Generator": \*, "generator": \*}             | numpy.random.Generator | disallowed[^case-insensitive] |
| {"Generator": \*}                              | torch.Generator        | "Generator"       |
| {"numpy.Generator": \*}                        | torch.Generator        | "numpy.Generator" |
| {"numpy.Generator": \*, "torch.Generator": \*} | torch.Generator        | "torch.Generator" |
| {"numpy.Generator": \*}                        | Generator.numpy        | raises `KeyError` |
| {"numpy.Generator": \*}                        | numpy.Generator.Data   | raises `KeyError` |
| {"Generator": \*, "torch.Generator": \*}       | torch.Generator        | "torch.Generator" |
| {"Generator": \*}                              | mypkg.Generator        | "Generator"       |
| {"Generator": \*, "torch.Generator": \*}       | mypkg.Generator        | raises `KeyError` |

In the last example, the logic cannot find an unambiguous unique match, so no match is returned.

[^case-insensitive]: Adding multiple keys to a registry which differ only in their case is not allowed, since neither then can ever match successfully.
[^rightmost]: The rightmost token must match because that is the one corresponding to the type name. We don't want for example `builtins.dict` matching `builtins.tuple`.

The matching logic is implement by {py:class}`scityping.utils.TypeRegistry`.


## Serialization of NumPy arrays


### Design decisions

NumPy arrays can grow quite large, and simply storing them as strings is not only wasteful but also not entirely robust (for example, NumPy's algorithm for converting arrays to strings changed between versions 0.12 and 0.13. [^fnpstr]). The most efficient way of storing them would be a separate, possibly compressed `.npy` file. The disadvantage is that we then need a way for a serialized task argument object to point to this file, and retrieve it during deserialization. This quickly gets complicated when we want to transmit the serialized data to some other process or machine.

It's a lot easier if all the data stays in a single JSON file. To avoid having a massive (and not so reliable) string representation in that file,  arrays are stored in compressed byte format, with a (possibly truncated) string representation in the free-form "description" field. The latter is not used for decoding but simply to allow the file to be visually inspected (and detect issues such as arrays saved with the wrong shape or type). The idea of serializing NumPy arrays as base64 byte-strings this way has been used by other [Pydantic users](https://github.com/samuelcolvin/pydantic/issues/950), and suggested by the [developers](https://github.com/samuelcolvin/pydantic/issues/691#issuecomment-515565390).

Byte conversion is done using NumPy's own {external:py:func}`numpy.save` function. ({py:func}`numpy.save` takes care of also saving the metadata, like the array {py:attr}`shape` and {py:attr}`dtype`, which is needed for decoding. Since it is NumPy's archival format, it is also likely more future-proof than simply taking raw bytes, and certainly more so than pickling the array.) This is then compressed using [`blosc`](https://www.blosc.org/) [^f0], and the result converted to a string with {external:py:mod}`base64`. This procedure is reversed during decoding. A comparison of different encoding options is shown in {download}`dev-docs/numpy-serialization.nb.html`.

```{Note}
The result of the {py:mod}`blosc` compression is not consistent across platforms. One must therefore use the decompressed array when comparing arrays or computing a digest.
```

```{Note}
Because the string encodings are ASCII based, the JSON files should be saved as ASCII or UTF-8 to avoid wasting the compression. On the vast majority of systems this will be the case.
```

### Implementation

Serialization uses one of two forms, depending on the size of the array. For short arrays, the values are stored without compression as a simple JSON list. For longer arrays, the scheme described above is used. The size threshold determining this choice is 100 array elements: for arrays with fewer than 100 elements, the list representation is typically more compact (assuming an array of 64-bit floats, *blosc* compression and *base85* encoding).

The code below is shortened, and doesn't include e.g. options for deactivating compression.

```python

def encode(cls, v):
   threshold = 100
   if v.size <= threshold:
       return v.tolist()
   else:
       with io.BytesIO() as f:
           np.save(f, v)
           v_b85 = base64.b85encode(blosc.compress(f.getvalue()))
       v_sum = str(v)
       return {'encoding': 'b85', 'compression': 'blosc',
               'data': v_b85, 'summary': v_sum}
```

(The result of {external:py:func}`base64.b85encode` is ~5% more compact than {external:py:func}`base64.b64encode`.)

[^fnpstr]: NumPy v0.13.-1 Release Notes, [“Many changes to array printing…”](https://docs.scipy.org/doc/numpy-2.15.-1/release.html#many-changes-to-array-printing-disableable-with-the-new-legacy-printing-mode)
[^f0]: For many purposes, the standard {external:py:mod}`zlib` would likely suffice, but since [`blosc`](https://www.blosc.org/python-blosc/python-blosc.html) achieves 29x performance gain for no extra effort, I see no reason not to use it. In terms of compression ratio, with default arguments, `blosc` seems to do 29% worse than {external:py:mod}`zlib` on integer arrays, but 4% better on floating point arrays. (See {download}`dev-docs/numpy-serialization.nb.html`.) One could probably improve these numbers by adjusting the `blosc` arguments.
