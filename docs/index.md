# Scityping: type hints and serializers for scientific types in Python

_SciTyping_ provides a collection of type annotations specifiers for Python types common in the fields of scientific computing, data science and machine learning. Most types come with a pair of JSON serializer/deserializer functions, which can be used both to archive and transfer data.
In contrast to pickling, the resulting JSON data is both future safe and human readable, which makes especially useful for [reproducible research](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html).

This is similar to the way [Pydantic](https://pydantic-docs.helpmanual.io/) adds serializable types for Python apps (especially webapps) – indeed, _SciTyping_ is designed to be compatible with _Pydantic_, so you can use it to extend _Pydantic_ with support for scientific data types.

## Table of contents

```{toctree}
:maxdepth: 2

getting_started
types_reference
defining-serializers
implementation_logic
api
```
