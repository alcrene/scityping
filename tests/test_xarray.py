from scityping import Serializable
from scityping.pydantic import BaseModel

import json
import numpy as np
import xarray as xr
import blosc  # Required for compressor to find it
import scityping
from scityping.xarray import DataArray, Dataset

import pytest

# NB: If the SciPy backend is used, serialization is done to NetCDF 3.
#     Some types (like uint32) are then coerced because that format doesn’t support them
#     List of coerced types: xr.backends.netcdf3._nc3_dtype_coercions
#     There are two reasons for xarray to use SciPy’s NetCDF support:
#     - neither NetCDF4 nor h5netcdf are installed
#     - xarray is serializing to bytes in-memory, rather than to disk.
#       (the NetCDF4 functions only support writing to disk)
#       We only write to disk if a path is given to `config.raw_bytes_directory`
#     Some tests check that NetCDF4 is used when possible; these tests will fail
#     if neither NetCDF4 no h5netcdf are installed.


@pytest.mark.parametrize("dtype", ["int32", "uint32", str, "float32", bool])
def test_dataarray(dtype, tmp_path):

    data = np.random.binomial(4, 0.2, size=10).astype(dtype)
    if dtype == str:
        # xarray will automatically compress to the smallest possible dtype when serializing
        # In order to succeed the test, we need to normalize/compress the initial array as well
        dtype = f"<U{max(len(a) for a in data)}"
        data = data.astype(dtype)

    np.random.seed(333)
    da = xr.DataArray(data, coords={"x": np.arange(10, dtype='int32')})  # xarray automatically compacts the coords dtype to 'int32' when it serializes
    da.name = "My data array"
    da.attrs = {"My attr": 3}
    da2 = Serializable.validate(Serializable.reduce(da))
    assert repr(da) == repr(da2)  # Check that they match exactly, including dtype

    # Ensure that short-circuit path using __scityping_from_base_type_ doesn’t drop attributes
    assert repr(da) == repr(Serializable.validate(da))

    class Model(BaseModel):
        da: DataArray

    model = Model(da=da)

    # Serializing a uint32 in-memory does not work (because SciPy only uses NetCDF3)
    # (it gets converted to int32)
    if dtype != "uint32":
        model2 = Model.parse_raw(model.json())

    # Serializing in-memory does not support Unicode characters (only Latin-1)
    scityping.config.annex_directory = None
    da_greek = da.rename({"x": "μ"})
    model_greek = Model(da=da_greek)
    with pytest.raises(ValueError):
        model_greek.json()

    # Once we set a path for data annexes, NetCDF4 is used
    scityping.config.annex_directory = tmp_path
    json_data = model_greek.json()
    model2 = Model.parse_raw(json_data)
    assert(repr(model_greek.da) == repr(model2.da))  # repr is used to check that also dtypes, attrs match

    digest = json.loads(json_data)["da"][1]["data"][1][1]["digest"]
    da_loaded = xr.load_dataarray(tmp_path/f"DataArray_{digest}.nc")
    assert(repr(model_greek.da) == repr(da_loaded))

@pytest.mark.parametrize("dtype", ["int32", "uint32", str, "float32", bool])
def test_dataset(dtype, tmp_path):

    alpha = np.arange(20).reshape(5,4)
    beta = np.arange(10).reshape(5,2)

    if dtype == str:
        # xarray will automatically compress to the smallest possible dtype when serializing
        # In order to succeed the test, we need to normalize/compress the initial array as well
        alpha = alpha.astype(f"<U{max(len(a) for a in alpha.astype(dtype).flat)}")
        beta = beta.astype(f"<U{max(len(a) for a in beta.astype(dtype).flat)}")
    elif dtype == bool:
        alpha = (alpha % 2).astype(bool)
        beta = (beta % 2).astype(bool)
    else:
        alpha = alpha.astype(dtype)
        beta = beta.astype(dtype)

    ds = xr.Dataset({
        "alpha": xr.DataArray(alpha,
                              coords={"x": 0.1*np.arange(5, dtype=np.float32),
                                      "y": 0.3*np.arange(4, dtype=np.float64)}
                               ),
        "beta" : xr.DataArray(beta,
                              coords={"x": 0.1*np.arange(5, dtype=np.float32),
                                      "z": 0.5*np.arange(2, dtype='int32')}
                               ),
    })
    ds.attrs = {"My attr": 3}

    ds2 = Serializable.validate(Serializable.reduce(ds))
    assert repr(ds) == repr(ds2)  # Check that they match exactly, including dtype

    # Ensure that short-circuit path using __scityping_from_base_type_ doesn’t drop attrs
    assert repr(ds) == repr(Serializable.validate(ds))

    class Model(BaseModel):
        ds: Dataset

    model = Model(ds=ds)

    # Serializing a uint32 in-memory does not work (because SciPy only uses NetCDF3)
    # (it gets converted to int32)
    if dtype != "uint32":
        model2 = Model.parse_raw(model.json())

    # Serializing in-memory does not support Unicode characters (only Latin-1)
    scityping.config.annex_directory = None
    ds_greek = ds.rename({"x": "μ"})
    model_greek = Model(ds=ds_greek)
    with pytest.raises(ValueError):
        model_greek.json()

    # Once we set a path for data annexes, NetCDF4 is used
    scityping.config.annex_directory = tmp_path
    json_data = model_greek.json()
    model2 = Model.parse_raw(json_data)
    assert(repr(model_greek.ds) == repr(model2.ds))  # repr is used to check that also dtypes, attrs match

    digest = json.loads(json_data)["ds"][1]["data"][1][1]["digest"]
    ds_loaded = xr.load_dataset(tmp_path/f"Dataset_{digest}.nc")
    assert(repr(model_greek.ds) == repr(ds_loaded))
