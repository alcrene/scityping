import scityping.scipy as ssp
import pytest

def test_distribution():
    # Distribution enforces that subclasses also subclass Distribution.Data
    with pytest.raises(TypeError):
        class MyDist(ssp.Distribution):
            class Data:
                dist: str
                args: tuple
                kwds: dict
