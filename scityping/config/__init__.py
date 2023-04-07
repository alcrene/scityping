from pathlib import Path
from typing import ClassVar
from pydantic import Field
from valconfig import ValConfig

# These are added to the namespace when deserializing a function
import numpy as np
import math

class Config(ValConfig):
    __default_config_path__: ClassVar=Path(__file__).parent/"defaults.cfg"

    trust_all_inputs: bool=False
    safe_packages: set={"__main__", "scityping"}
    default_namespace: dict=Field(
        default_factory=lambda: {'__builtins__': __builtins__,
                                 'np': np,
                                 'math': math})

config = Config()
