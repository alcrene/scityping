import contextlib
import os
import functools
import sys
import hashlib
import tempfile
from typing import Union, Tuple

import xarray as xr

from .config import config
from .base import Serializable
from .base_types import SerializedData
from .numpy import (encoders, decoders, compressors, decompressors,
                    _EncoderType, _CompressionType)

# TODO: Reduce duplication between DataArray & Dataset (and evtl. numpy.Array).
#       The only line which differs between the two is the last line of `decode`.
# TODO: Allow writing to a separate file (how to manage separate output files needs to be decided at the project level)
#       Using `.to_netcdf()` without args to return a bytes object only works with scipy, and therefore only with the NetCDF3 format.

# NB: xarray objects already provide a serializer `.to_netcdf()` which converts everything to a bytes sequence.
#     So all we need to do is wrap this with a compressor and encoder, same as we do for NumPy arrays
#     HOWEVER, when writing to bytes, xarray falls back to the 'scipy' engine, which only implements NetCDF3.
#     This means that certain types like uint are not supported.

class _AnnexData(SerializedData):
    """
    This format uses the object’s own `to_netcdf` method to write an external
    file in a annex directory.
    Advantages: Allows the use of NetCDF4. More memory efficient. Variable or
      file is easier to inspect since it doesn’t contain the data blob.
    Disadvantage: Annex file must be kept with the serialized data, otherwise
      the data cannot be reloaded. Only works with Python 3.11+
    The `summary` attribute is not actually used by `scityping` itself; it allows
    a human inspecting the output to check the array shape, dtype, etc.
    """
    # TODO?: If we could hash the DataArray before writing, we could check whether a
    #   write is necessary at all. We also would remove the requirement of Python 3.11+.
    #   One possibility would be to use Dask’s tokenize() function, but that would
    #   add a dependency on Dask.
    clsname = None  # Overwrite in subclasses
    summary: str
    digest: str
    @classmethod
    def encode(cls, da: Union[xr.DataArray,xr.Dataset]):
        fi, path = tempfile.mkstemp(dir=config.annex_directory)  # NB: Putting temp file in the target directory ensures that the `os.rename` below is not across devices
        da.to_netcdf(path)
        with open(fi, 'rb') as f:
            # NB: hashlib.file_digest was added in Python 3.11
            digest = hashlib.file_digest(f, lambda: hashlib.sha1(usedforsecurity=False)
                ).hexdigest()
        annex_name = cls.get_annex_name(digest)
        config._annex_files.append(annex_name)
        os.rename(path, config.annex_directory/annex_name)
        
        summary = str(da) if config.include_summaries else ""
        return cls(summary, digest)

    @classmethod
    def decode(cls, data):
        annex_file = cls.get_annex_name(data.digest)
        if config.annex_directory is None:
            raise FileNotFoundError("Attempted to load a {cls.clsname} file with "
                "an annex, but no annex directory is set. Please set "
                "`config.annex_directory` to the directory containing the "
                f"file '{annex_file}'.")
        return cls.loader(config.annex_directory/annex_file)

    @classmethod
    def get_annex_name(cls, digest: str) -> str:
        return f"{cls.clsname}_{digest}.nc"

    # Must be defined in derived class:
    # - clsname
    # - decode()

class AnnexDataArrayData(_AnnexData):
    clsname = "DataArray"
    loader = xr.load_dataarray

class AnnexDatasetData(_AnnexData):
    clsname = "Dataset"
    loader = xr.load_dataset

class _InlineData(SerializedData):
    """
    This format converts a DataArray or Dataset to a bytes sequence,
    compresses the sequence, then encodes into to plain text.
    Advantages: Everything stays in one variable or file.
    Disadvantages: Only supports NetCDF3. Encoding to plain text inflates the size
      by about 30%. The large data blob makes the variable or file difficult to inspect.
    The `summary` attribute is not actually used by `scityping` itself; it allows
    a human inspecting the output to check the array shape, dtype, etc.
    """
    clsname = None  # Overwrite in subclasses
    encoding: _EncoderType
    compression: _CompressionType
    summary: str
    data: bytes
    @classmethod
    def encode(cls, xrobj,
               compression: Union[str,Tuple[str,...]]=("blosc","zlib"),
               encoding="b85"):
        # Parse arguments
        if compression is None:
            compression = 'none'
        if isinstance(compression, (tuple, list)):
            for comp in compression:
                compressor = compressors.get(comp)
                if compressor:
                    compression = comp  # Value of `compression` is stored with the data
                    break
            else:
                raise ModuleNotFoundError("None of the specified compressors "
                                          f"were found: {compression}.")
        else:
            compressor = compressors[compression]
        encoder = encoders[encoding]
        # Convert array to bytes
        try:
            v_bytes = xrobj.to_netcdf()
        except ValueError as e:
            raise ValueError(
                f"Unable to perform an in-memory serialization of an xarray {cls.clsname}. "
                "Note that xarray falls back to SciPy’s implementation when serializing "
                "in-memory, which only implements netCDF3. To use the netCDF4 format, "
                "first ensure you are running Python 3.11 or later, then set "
                "the value of `scityping.config.annex_directory` to a valid path. "
                f"This will cause {cls.clsname}s to be written to that directory; the "
                "returned serialized array will only contain a path to the data."
                )
        # Compress and encode the bytes
        array_encoded = encoder(compressor(v_bytes))
        # Return
        summary = str(xrobj) if config.include_summaries else ""
        return cls(encoding, compression, summary, array_encoded)
    @classmethod
    def decode(cls, data):
        decoder = decoders[data.encoding]
        decompressor = decompressors[data.compression]
        v_bytes = decompressor(decoder(data.data))
        return cls.loader(v_bytes)

class InlineDataArrayData(_InlineData):
    clsname = "DataArray"
    loader = xr.load_dataarray

class InlineDatasetData(_InlineData):
    clsname = "Dataset"
    loader = xr.load_dataset


# TODO?: Implement fallback to plaintext, for small DataArray (analogous to numpy.ListArrayData)

class DataArray(Serializable, xr.DataArray):
    """
    Stores `xarray.DataArray` objects by converting them to bytes with their
    `.to_netcdf()` method.
    """
    @classmethod
    def __scityping_from_base_type__(cls, da:xr.DataArray) -> "DataArray":
        """
        Convert a plain xr.DataArray into a scityping one.
        When this function is not available, scityping falls back to doing
        `subcls.validate(subcls.Data.encode(value))`
        For an xarray, this may involve unnecessarily writing and reading to disk;
        we avoid this by providing this function.
        """
        return cls(da, coords=da.coords, attrs=da.attrs, name=da.name)

    class Data(SerializedData):
        data: Union[AnnexDataArrayData,InlineDataArrayData]
        @classmethod
        def encode(cls, da: xr.DataArray):
            if config.annex_directory and sys.version_info >= (3, 11):
                with contextlib.suppress(ValueError):
                    return {"data": AnnexDataArrayData.encode(da)}
            return {"data": InlineDataArrayData.encode(da)}
        @staticmethod
        def decode(data):
            return data.data.decode(data.data)


# NB: Dataset also provides `.to_netcdf()`, so we can use exactly the same implementation for Dataset.
class Dataset(Serializable, xr.Dataset):
    """
    Stores `xarray.DataArray` objects by converting them to bytes with their
    `.to_netcdf()` method.
    """
    @classmethod
    def __scityping_from_base_type__(cls, ds:xr.Dataset) -> "Dataset":
        """
        Convert a plain xr.Dataset into a scityping one.
        When this function is not available, scityping falls back to doing
        `subcls.validate(subcls.Data.encode(value))`
        For an xarray, this may involve unnecessarily writing and reading to disk;
        we avoid this by providing this function.
        """
        return cls(ds, coords=ds.coords, attrs=ds.attrs)

    class Data(SerializedData):
        data: Union[AnnexDatasetData,InlineDatasetData]
        @classmethod
        def encode(cls, da: xr.Dataset):
            if config.annex_directory and sys.version_info >= (3, 11):
                with contextlib.suppress(ValueError):
                    return {"data": AnnexDatasetData.encode(da)}
            return {"data": InlineDatasetData.encode(da)}
        @staticmethod
        def decode(data):
            return data.data.decode(data.data)
