from pathlib import Path
from typing import ClassVar
from pydantic import Field
from .validating_config import ValidatingConfig

# These are added to the namespace when deserializing a function
import numpy as np
import math

class Config(ValidatingConfig):
    default_config_file: ClassVar=Path(__file__).parent/"defaults.cfg"
    config_module_name : ClassVar = __name__
    ensure_user_config_exists: ClassVar = False

    trust_all_inputs: bool=False
    safe_packages: set={"__main__", "scityping"}
    default_namespace: dict=Field(
        default_factory=lambda: {'__builtins__': __builtins__,
                                 'np': np,
                                 'math': math})

config = Config()
