# Version 0.7.0

- BREAKING CHANGE: `SerializedData` now used a stdlib dataclass, instead of a Pydantic dataclass.
- Basic deserialization capabilities for stdlib dataclasses.
- `config` now uses a stdlib dataclass
- Remove hard dependency on Pydantic
- Full overhaul of documentation