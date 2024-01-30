from scityping import Serializable
from scityping.pydantic import BaseModel

import numpy as np
import xarray as xr
import blosc  # Required for compressor to find it
from scityping.xarray import DataArray, Dataset

# NB: `.to_netcdf()` converts int64 values to int32 (at least where possible)
#     Therefore to get exactly identical string representation in the deserialized
#     values, we need to use int32 for the data. (Or a float dtype, since float precision isn’t changed.)

def test_dataarray():

    # NB: If the SciPy backend is used, serialization is done to NetCDF 3.
    #     Some types (like uint32) are then coerced because that format doesn’t support them
    #     List of coerced types: xr.backends.netcdf3._nc3_dtype_coercions
    #     For this test we stick with types supported by NetCDF3
    arr = xr.DataArray(np.arange(3, dtype="int32"))
    arr2 = Serializable.validate(Serializable.reduce(arr))
    assert repr(arr) == repr(arr2)  # Check that they match exactly, including dtype

    class Model(BaseModel):
        da: DataArray

    model = Model(da=arr)
    model2 = Model.parse_raw(model.json())

    assert(repr(model.da) == repr(model2.da))  # repr is used to check that also dtypes, attrs match

def test_dataset():

    # NB: If the SciPy backend is used, serialization is done to NetCDF 3.
    #     Some types (like uint32) are then coerced because that format doesn’t support them
    #     List of coerced types: xr.backends.netcdf3._nc3_dtype_coercions
    #     For this test we stick with types supported by NetCDF3

    ds = xr.Dataset({
        "alpha": xr.DataArray(np.arange(20, dtype=np.int32).reshape(5,4),
                              coords={"x": 0.1*np.arange(5, dtype=np.float32),
                                      "y": 0.3*np.arange(4, dtype=np.float64)}
                               ),
        "beta" : xr.DataArray(np.arange(10, dtype=np.int32).reshape(5,2),
                              coords={"x": 0.1*np.arange(5, dtype=np.float32),
                                      "z": 0.5*np.arange(2)}
                               ),
    })

    ds2 = Serializable.validate(Serializable.reduce(ds))
    assert repr(ds) == repr(ds2)  # Check that they match exactly, including dtype

    class Model(BaseModel):
        ds: Dataset

    model = Model(ds=ds)
    model2 = Model.parse_raw(model.json())

    assert(repr(model.ds) == repr(model2.ds))  # repr is used to check that also dtypes, attrs match