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
