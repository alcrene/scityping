import pytest
import logging
import scityping.scipy as stsp
from scityping.pydantic import BaseModel

# TODO: Move tests from statGLOW.stats.tests.test_serialization to here

def test_distribution():
    # Distribution enforces that subclasses also subclass Distribution.Data
    with pytest.raises(TypeError):
        class MyDist(stsp.Distribution):
            class Data:
                dist: str
                args: tuple
                kwds: dict

def test_distributions(caplog):
    from scipy import stats
    from scityping.scipy import (Distribution, UniDistribution,
                                 MvDistribution, MvNormalDistribution)

    for D in [stats.poisson(3.1), stats.norm(-1, scale=3), stats.zipf(a=2.2)]:
        D.random_state = 123
        # By default the RNG is serialized, so both D and D2 return the same values
        with caplog.at_level(logging.ERROR, logger="scityping.scipy"):  # Hide warnings about serializing arguments
            D2 = Distribution.validate(UniDistribution.reduce(D))
        assert (D.rvs(25) == D2.rvs(25)).all()
        # The RNG state is generally the largest amount of data (especially for MT19337),
        # so we can skip serializing if we don't need it.
        with caplog.at_level(logging.ERROR, logger="scityping.scipy"):  # Hide warnings about serializing arguments
            D3 = Distribution.validate(UniDistribution.reduce(D, include_rng_state=False))
        assert (D.rvs(25) != D3.rvs(25)).any()  # Now the generated random numers are different

    # For multivariate distributions, in contrast to univariate ones, a different
    # subclass of `Distribution` is needed for each one.
    D = stats.multivariate_normal([2, -2], [[2.2, 1.],[1., 1.3]])
    D.random_state = 321
    D2 = Distribution.validate(MvNormalDistribution.reduce(D))
    assert (D.rvs(25) == D2.rvs(25)).all()
    D3 = Distribution.validate(MvNormalDistribution.reduce(D, include_rng_state=False))
    assert (D.rvs(5) != D3.rvs(5)).all()  # With R-valued distributions we can be more aggressive in the test and use .all() instead of .any()

    # Not all multivariate distributions are implemented yet
    D = stats.dirichlet([.2, .7])
    with pytest.raises(NotImplementedError):
        MvDistribution.reduce(D)

    # Test complete serialization to JSON
    class Foo(BaseModel):
        dist: Distribution
    for D in [stats.poisson(3.1), stats.norm(-1, scale=3), stats.zipf(a=2.2)]:
        foo = Foo(dist=D)
        foo2 = Foo.parse_raw(foo.json())


