import logging
import dataclasses
import pytest

import numpy as np
import pint
from pydantic import validator
from pydantic import ValidationError
from scityping.pydantic import BaseModel, dataclass

from typing import List, Tuple
from scityping.numpy import NPValue, Array, NPGenerator, RandomState
from scityping.pint import PintUnit, PintValue

# TODO: Systematically test casting of NPValue and Array, especially with
#       generic types
# TODO: Test Torch types

def test_numpy(caplog):

    class Model(BaseModel):
        a: float
        # dt: float=0.01
        dt: float
        def integrate(self, x0, T, y0=0):
            x = x0; y=y0
            a = self.a; dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=a*x*dt
            return x
        @property
        def params(self):
            return ParameterSet(self.dict())

    m = Model(a=-0.5, dt=0.01)
    assert repr(m) == "Model(a=-0.5, dt=0.01)"
    assert np.isclose(m.integrate(1, 5.), -0.924757713688715)
    assert type(m.a) is float

    class Model2(Model):
        b: NPValue[np.float64]
        w: Array[np.float32]
        def integrate(self, x0, T, y0=0):
            x = w*x0; y = w*y0
            α = np.array([self.a, self.b]); dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=α*x*dt
            return x.sum()

    # Test subclassed List and Tuple
    # TODO: tuples with floats
    class ModelB(BaseModel):
        l1: list
        l2: List[int]
        t1: tuple
        # t2: Tuple[np.float16, np.float32]
        pass

    mb = ModelB(l1=['foo', 33], l2=[3.5, 3], t1=(1, 2, 3, 4, 5))
    assert mb.l1 == ['foo', 33]
    assert mb.l2 == [3, 3]
    assert mb.t1 == (1, 2, 3, 4, 5)

    # NOTE: Some of the tests below do not work presently with
    #       dataclasses, due to the way keyword vs non-keyword parameters
    #       are ordered (see discussion https://bugs.python.org/issue36077)
    #       Pydantic models however allows setting default/computed
    #       override base class attributes, without having to use a default for
    #       new every attribute in the derived class
    class Model3(Model):
        a: NPValue[float]  = 0.3
        β: NPValue[float] = None
        @validator('β', pre=True, always=True)
        def set_β(cls, v, values):
            a, dt = (values.get(x, None) for x in ('a', 'dt'))
            # This test is important, because if `a` or `dt` are missing
            # (as when we assign to `m4_fail` below) or trigger a validation
            # error, they will be missing from `values` and raise KeyError here,
            # (See https://pydantic-docs.helpmanual.io/usage/validators)
            if all((a, dt)):
                return a*dt
        def integrate(self, x0, T, y0=0):
            x = x0; y=y0
            β = self.β; dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=β*x
            return x

    # NOTE: Following previous note: with vanilla dataclasses Model4 would need
    #       to define defaults for every attribute.
    class Model4(Model3):
        b: NPValue[np.float32]
        w: Array[np.float32] = (np.float32(1), np.float32(0.2))
        γ: Array[np.float32] = None
        @validator('γ', pre=True, always=True)
        def set_γ(cls, v, values):
            a, b, β = (values.get(x, None) for x in ('a', 'b', 'β'))
            if all((a, b, β)):
                return β * np.array([1, b/a], dtype='float32')
        def integrate(self, x0, T, y0=0):
            x = w*x0; y = w*y0
            γ = self.γ; dt = self.dt
            for t in np.arange(0, T, self.dt):
                x += y*dt; y+=γ*x
            return x.sum()

    # Basic type conversion
    w64 = np.array([0.25, 0.75], dtype='float64')
    w32 = w64.astype('float32')
    w16 = w64.astype('float16')
    with pytest.raises(ValidationError):   # With vanilla dataclass: TypeError
        largeb = 57389780668102097303  # int(''.join((str(i) for i in np.random.randint(10, size=20))))
        m2_np = Model2(a=-0.5, b=largeb, w=w16, dt=0.01)
    m2_np = Model2(a=-0.5, b=-0.1, w=w16, dt=0.01)
    assert repr(m2_np) == "Model2(a=-0.5, dt=0.01, b=-0.1, w=array([0.25, 0.75], dtype=float32))"
    assert m2_np.b.dtype is np.dtype('float64')
    # Test conversion from ints and list
    m2_nplist = Model2(a=-0.5, b=np.float32(-0.1), w=[np.float32(0.25), np.float32(0.75)], dt=0.01)
    assert m2_nplist.w.dtype is np.dtype('float32')

    # Test computed fields and numpy type conversions
    m3 = Model3(a=.25, dt=0.02)
    assert type(m3.a) != type(m.a)
    assert type(m3.a) is np.dtype(float).type
    assert type(m3.β) is np.dtype(float).type
    assert m3.β == 0.005
    # These two constructors seem to specify the same model (default w is
    # simply reproduced), but because validators are not called on default
    # arguments in the first case the default w remains a tuple.
    # You can use the following validator to force its execution
    # @validator('w', always=True)
    #     def set_w(cls, v):
    #         return v
    m4_default = Model4(a=.25, dt=0.02, b=np.float32(0.7))
    m4         = Model4(a=.25, dt=0.02, b=np.float32(0.7),
                        w=(np.float32(1), np.float32(0.2)))
    assert isinstance(m4_default.w, tuple)
    assert isinstance(m4.w, np.ndarray) and m4.w.dtype.type is np.float32

    m4_16 = Model4(a=.25, dt=0.02, b=np.float32(0.3), w=w16)
    m4_32 = Model4(a=.25, dt=0.02, b=np.float32(0.3), w=w32)
    assert m4_16.w.dtype == 'float32'
    assert m4_32.w.dtype == 'float32'
    # Casts which raise TypeError
    with pytest.raises(ValidationError):   # With stdlib dataclass: TypeError
        m4_fail = Model4(a=.3, b=0.02, w=w64)
    with pytest.raises(ValidationError):   # With stdlib dataclass: TypeError
        wint = np.array([1, 2])
        m2_npint = Model2(a=-0.5, b=-0.1, w=wint, dt=0.01)

    # Import after defining Model to test difference in mapping of `float` type

def test_rng():
    class RandomModel(BaseModel):
        rng: NPGenerator

    # # FIXME
    # RandomModel.schema()

    from numpy.random import Generator, PCG64, MT19937, SFC64, Philox
    seed = 953235987

    rm_pcg = RandomModel(rng=Generator(PCG64(seed)))
    rm_mt  = RandomModel(rng=Generator(MT19937(seed)))
    rm_sfc = RandomModel(rng=Generator(SFC64(seed)))
    rm_phi = RandomModel(rng=Generator(Philox(seed)))

    # Save models in their current initialized state
    pcg_json = rm_pcg.json()
    mt_json  = rm_mt.json()
    sfc_json = rm_sfc.json()
    phi_json = rm_phi.json()

    # Draw from models, advancing the bit generator
    pcg_draws = rm_pcg.rng.random(size=5)
    mt_draws  = rm_mt.rng.random(size=5)
    sfc_draws = rm_sfc.rng.random(size=5)
    phi_draws = rm_phi.rng.random(size=5)

    # Create new model copies, in their original initialized states
    rm_pcg2 = RandomModel.parse_raw(pcg_json)
    rm_mt2  = RandomModel.parse_raw(mt_json)
    rm_sfc2 = RandomModel.parse_raw(sfc_json)
    rm_phi2 = RandomModel.parse_raw(phi_json)

    # Draw again => same numbers as before
    assert np.all(pcg_draws == rm_pcg2.rng.random(size=5))
    assert np.all(mt_draws  == rm_mt2.rng.random(size=5))
    assert np.all(sfc_draws == rm_sfc2.rng.random(size=5))
    assert np.all(phi_draws == rm_phi2.rng.random(size=5))

    # Drawing _again_ produces different numbers => these really are random numbers
    assert np.all(pcg_draws != rm_pcg2.rng.random(size=5))
    assert np.all(mt_draws  != rm_mt2.rng.random(size=5))
    assert np.all(sfc_draws != rm_sfc2.rng.random(size=5))
    assert np.all(phi_draws != rm_phi2.rng.random(size=5))

def test_legacy_rng():
    class RandomModel(BaseModel):
        rng: RandomState

    # # FIXME
    # RandomModel.schema()

    seed = 953235987
    rm_leg = RandomModel(rng=RandomState(seed))

    # Save models in their current initialized state
    leg_json = rm_leg.json()

    # Draw from models, advancing the bit generator
    leg_draws = rm_leg.rng.random(size=5)

    # Create new model copies, in their original initialized states
    rm_leg2 = RandomModel.parse_raw(leg_json)

    # Draw again => same numbers as before
    assert np.all(leg_draws == rm_leg2.rng.random(size=5))

    # Drawing _again_ produces different numbers => these really are random numbers
    assert np.all(leg_draws != rm_leg2.rng.random(size=5))

def test_distributions(caplog):
    from scipy import stats
    from scityping.scipy import (Distribution, UniDistribution,
                                 MvDistribution, MvNormalDistribution)

    for D in [stats.poisson(3.1), stats.norm(-1, scale=3), stats.zipf(a=2.2)]:
        D.random_state = 123
        # By default the RNG is serialized, so both D and D2 return the same values
        with caplog.at_level(logging.ERROR, logger="scityping.scipy"):  # Hide warnings about serializing arguments
            D2 = Distribution.validate(UniDistribution.json_encoder(D))
        assert (D.rvs(25) == D2.rvs(25)).all()
        # The RNG state is generally the largest amount of data (especially for MT19337),
        # so we can skip serializing if we don't need it.
        with caplog.at_level(logging.ERROR, logger="scityping.scipy"):  # Hide warnings about serializing arguments
            D3 = Distribution.validate(UniDistribution.json_encoder(D, include_rng_state=False))
        assert (D.rvs(25) != D3.rvs(25)).any()  # Now the generated random numers are different

    # For multivariate distributions, in contrast to univariate ones, a different
    # subclass of `Distribution` is needed for each one.
    D = stats.multivariate_normal([2, -2], [[2.2, 1.],[1., 1.3]])
    D.random_state = 321
    D2 = Distribution.validate(MvNormalDistribution.json_encoder(D))
    assert (D.rvs(25) == D2.rvs(25)).all()
    D3 = Distribution.validate(MvNormalDistribution.json_encoder(D, include_rng_state=False))
    assert (D.rvs(5) != D3.rvs(5)).all()  # With R-valued distributions we can be more aggressive in the test and use .all() instead of .any()

    # Not all multivariate distributions are implemented yet
    D = stats.dirichlet([.2, .7])
    with pytest.raises(NotImplementedError):
        MvDistribution.json_encoder(D)


def test_pint():
    # NB: Pint v0.18 changed to 'ApplicationRegistry'; before was 'LazyRegistry'
    #     The second `getattr` is in case LazyRegistry is removed in the future
    ApplicationRegistry = getattr(pint, 'ApplicationRegistry',
                                  getattr(pint, 'LazyRegistry', None))

    # PintUnitMeta
    assert PintUnit._ureg is None
    ureg = PintUnit.ureg
    assert isinstance(ureg, ApplicationRegistry)
    assert PintUnit._ureg is ureg
    assert PintUnit.ureg is ureg
    with pytest.raises(TypeError):
        PintUnit.ureg = 3
    PintUnit.ureg = ureg

    # PintUnit
    assert PintUnit.validate(ureg.m) == ureg.m
    assert PintUnit.validate(1.*ureg.m) == ureg.m  # Also accept Quantities, iff they have magnitude 1
    with pytest.raises(ValueError):
        PintUnit.validate(2.*ureg.m)
    u = ureg.m
    u_json = PintUnit.json_encoder(u)
    assert u_json == ('scityping.pint.PintUnit', PintUnit.Data(name='meter'))
    assert PintUnit.validate(u_json) == ureg.m

    uu = PintUnit(u)
    assert uu.apply(5) == 5*ureg.m == 5*uu
    assert uu.apply(10*ureg.m) == 10*ureg.m == uu.apply(10*uu) == 10*uu
    v = uu.apply(1000*ureg.cm)  # Compatible units get convert
    assert v.units == ureg.m
    assert v == 10*ureg.m
    with pytest.raises(ValueError):
        uu.apply(5*ureg.s)  # `apply` ensures the result has exactly the prescribed units
                            # (in contrast to multiplication, which would simply add those units)

    # PintValue
    assert PintValue.validate(3.*ureg.s) == 3.*ureg.s
    # with pytest.raises(TypeError):
    #     PintValue.validate(3.)
    assert PintValue.validate(3.*ureg.dimensionless) == PintValue(3.) == 3.*ureg.dimensionless
    v = 3.*ureg.s
    vms = v.to('ms')
    v_json = PintValue.json_encoder(v)
    vms_json = PintValue.json_encoder(vms)
    assert v_json == ('scityping.pint.PintQuantity', PintValue.Data(data=(3.0, (('second', 1),))))
    assert v_json != vms_json  # Serialization changes depending on units
    assert PintValue.validate(v_json) == PintValue.validate(vms_json)  # But deserialized values still compare equal

def test_torch():
    import torch
    from scityping.torch import TorchTensor, TorchModule, TorchGenerator
    from scityping.torch import torch_module_state_encoder, torch_module_state_decoder

    # Scalars
    s_u8   = torch.tensor(np.uint8(5))
    s_i32  = torch.tensor(np.int32(5))
    s_f64  = torch.tensor(np.float64(5))
    s_bool = torch.tensor(True)
    # Vectors
    v_u8   = torch.tensor(np.uint8  ([5, 8, 9]))
    v_i32  = torch.tensor(np.int32  ([5, 8, 9]))
    v_f64  = torch.tensor(np.float64([5, 8, 9]))
    v_bool = torch.tensor([True, False, True])
    # Matrices
    m_u8   = torch.outer(v_u8, v_u8)
    m_i32  = torch.outer(v_i32, v_i32)
    m_f64  = torch.outer(v_f64, v_f64)
    m_bool = torch.outer(v_bool, v_bool)

    for a in [s_u8, s_i32, s_f64, s_bool,
              v_u8, v_i32, v_f64, v_bool,
              m_u8, m_i32, m_f64, m_bool]:
        b = TorchTensor.validate(TorchTensor.json_encoder(a))
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert (a == b).all()

# test_numpy(None)
test_rng()