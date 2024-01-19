# from scityping import Serializable
from scityping.pydantic import BaseModel

import numpy as np
import xarray as xr
import blosc  # Required for compressor to find it
from scityping.xarray import DataArray

def test_dataarray():

    # NB: If the SciPy backend is used, serialization is done to NetCDF 3.
    #     Some types (like uint32) are then coerced because that format doesn’t support them
    #     List of coerced types: xr.backends.netcdf3._nc3_dtype_coercions
    #     For this test we stick with types supported by NetCDF3
    arr = xr.DataArray(np.arange(3, dtype="int32"))
    arr2 = DataArray.validate(DataArray.reduce(arr))
    assert repr(arr) == repr(arr2)  # Check that they match exactly, including dtype

    class Model(BaseModel):
        da: DataArray

    model = Model(da=arr)
    model2 = Model.parse_raw(model.json())

    assert(repr(model.da) == repr(model2.da))  # repr is used to check that also dtypes, attrs match