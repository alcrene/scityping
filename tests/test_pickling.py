import pickle
# import numpy as np
# import scipy.stats as stats
# import scityping.numpy as stnp
# import scityping.scipy as stsp
from scityping.functions import PureFunction
from scityping.config import config
from functools import partial

def test_pickling():

    old_config = dict(trust_all_inputs=config.trust_all_inputs)
    config.trust_all_inputs = True  # Allow deserializing functions

    @PureFunction
    def f(x, n):
        return x**n

    # NB: If we didn't support pickling, this would fail because f() is
    # defined in the local scope.
    fbytes = pickle.dumps(f)
    f2 = pickle.loads(fbytes)

    assert f2 is not f
    assert f2(4, 6) == f(4, 6)

    g = partial(f, n=3)

    gbytes = pickle.dumps(g)
    g2 = pickle.loads(gbytes)

    assert g2 is not g
    assert g2(6) == f(6, n=3)

    config.trust_all_inputs = old_config["trust_all_inputs"]
