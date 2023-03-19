"""
Proxy module for Pydantic field types, which provides fallbacks in case
Pydantic is not installed.
"""

try:
    from pydantic import StrictStr, StrictBytes, StrictInt, StrictFloat, StrictBool
except ModuleNotFoundError:
    StrictStr = str
    StrictBytes = bytes
    StrictInt = int
    StrictFloat = float
    StrictBool = bool
