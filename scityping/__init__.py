from .base import *
from .base_types import *

# Allow writing `scityping.config.trust_all_inputs = True`
# instead of    `scityping.config.config.trust_all_inputs = True`
from .config import config
