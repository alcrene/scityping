import scityping.scipy as stsp
import pytest

def test_distribution():
    # Distribution enforces that subclasses also subclass Distribution.Data
    with pytest.raises(TypeError):
        class MyDist(stsp.Distribution):
            class Data:
                dist: str
                args: tuple
                kwds: dict
