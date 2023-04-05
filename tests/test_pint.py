import logging
import dataclasses
import pytest

import numpy as np
import pint
from pydantic import validator
from pydantic import ValidationError
from scityping.pydantic import BaseModel, dataclass

from typing import List, Tuple
from scityping.pint import PintUnit, PintValue

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
    u_json = PintUnit.reduce(u)
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
    v_json = PintValue.reduce(v)
    vms_json = PintValue.reduce(vms)
    assert v_json == ('scityping.pint.PintQuantity', PintValue.Data(data=(3.0, (('second', 1),))))
    assert v_json != vms_json  # Serialization changes depending on units
    assert PintValue.validate(v_json) == PintValue.validate(vms_json)  # But deserialized values still compare equal
