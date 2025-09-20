from pathlib import Path
from collections.abc import Mapping
from typing import Any
from pydantic import field_validator, Field
from valconfig import ValConfig

# These are added to the namespace when deserializing a function
import numpy as np
import math
ModuleType = type(math)

class Config(ValConfig):
    """
    Attributes
    ----------
    trust_all_inputs: If True, allow unsafe deserializations from any source.
       This is a “turn it all off!” flag for those who can’t be bothered to
       set `safe_packages`. Default is False.
    safe_packages: List of packages which are expected to contain define
       types requiring elevated privileges to deserialize (typically an import).
       The idea is that if you installed package `foo`, then if you implicitely
       trust that the code inside `foo` is safe to execute. Therefore if some
       serialized data requires `import foo` to deserialize, you consider this
       safe, and allow `scityping` to proceed.
       Packages may add themselves (technically any package) to this list, to
       allow deserialization of their own types.
    default_function_deserialization_namespace:
       Namespace used when deserializing functions, if no other namespace
       is provided. The default value is
       ``{'__builtins__': __builtins__, 'np': np, 'math': math}``
    include_summaries: If True (default), certain types which export binary
       blobs may also include a string representation, to help human inspecting
       the serialized data. Setting to False may be useful if the result is
       intended to be hashed, since text representations are typically less
       stable across versions.
    annex_directory: If set, allows certain types to write their data in
       separate annex files. This has certain advantages: easier to read JSON
       files since binary blobs are not inlined in JSON files; less memory;
       more capable serialization engine (in the case of xarray). The main
       downside is that one needs to keep track of multiple files, and ensure
       that they move together. To help with this, `scityping` will update the
       global list `scityping.config.annex_files` every time it writes to the
       annex directory; it is up to the user to use this information.
    annex_files: Read-only; Internally maintained list of files written in
       `annex_directory`. See note below.


    Note
    ----
    `scityping` was not developed with security as a priority. The `safe_packages`
    flag is meant to prevent the most basic security holes, like executing ``eval``
    on unsanitized data. It is **not** meant to protect you from an attacker:
    `scityping`’s security model remains that it assumes that you are running
    your own code with input files created by you (or by someone you trust).
    `scityping` is especially robustness against unintential errors, but
    against adversarial ones it is similar to any other package installed
    from PyPI.

    If you intend to use `scityping` in a context where security is important,
    please consult with the relevant expertise, and consider running your program
    in a sandbox. (Which is generally good practice anyway.)

    Note
    ----
    `annex_files` is a globally accessible record of external annex
    filenames that have been saved by scityping. User code can use this to
    retrieve those files, or store their names as part of record keeping.
    Conceptually it is a 'state' variable rather than a 'config' variable:
    scityping will _write_ to `annex_files`, while users will _read_
    `annex_files`. (For 'config' variables it is the other way around.)
    Functionally both 'state' and 'config' require the same thing – a global
    singleton instance – which is why we reuse `config` for `annex_files`.

    """
    trust_all_inputs : bool=False
    safe_packages    : set={"__main__", "scityping"}
    default_function_deserialization_namespace: dict[str, ModuleType|Mapping] = \
        Field(default_factory=lambda: {'__builtins__': __builtins__, 'np': np, 'math': math})
        # NB: Using default_factory avoids Pydantic making a deepcopy of the value

    include_summaries: bool=True
    annex_directory  : Path|None=None

    # Global state variables:
    _annex_files: list=[]

    # TODO?: A slightly annoying thing is that when we print the Config,
    #       the  __builtins__ dict (b/c it is a dict and not a module)
    #       gets printed full, drowning everything else.
    #       Ideally we would replace it with a short string like `<builtins>`

    @property
    def annex_files(self):
        return self._annex_files[:]  # Make a copy so the original stays read-only

config = Config()
