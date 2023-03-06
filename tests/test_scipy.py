import scityping.scipy as stsp
import pytest

# TODO: Move tests from statGLOW.stats.tests.test_serialization to here

def test_distribution():
    # Distribution enforces that subclasses also subclass Distribution.Data
    with pytest.raises(TypeError):
        class MyDist(stsp.Distribution):
            class Data:
                dist: str
                args: tuple
                kwds: dict
