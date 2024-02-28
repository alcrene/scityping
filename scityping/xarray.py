import xarray as xr
from typing import Union, Tuple

from .base import Serializable
from .base_types import SerializedData
from .numpy import (encoders, decoders, compressors, decompressors,
                    _EncoderType, _CompressionType)

# TODO: Reduce duplication between DataArray & Dataset (and evtl. numpy.Array).
#       The only line which differs between the two is the last line of `decode`.
# TODO: Allow writing to a separate file (how to manage separate output files needs to be decided at the project level)
#       Using `.to_netcdf()` without args to return a bytes object only works with scipy, and therefore only with the NetCDF3 format.
# TODO: Use a proxy file object (like we do with NumPy arrays) so that we can use NetCDF4 instead of NetCDF4.

# NB: xarray objects already provide a serializer `.to_netcdf()` which converts everything to a bytes sequence.
#     So all we need to do is wrap this with a compressor and encoder, same as we do for NumPy arrays
#     HOWEVER, when writing to bytes, xarray falls back to the 'scipy' engine, which only implements NetCDF3.
#     This means that certain types like uint are not supported.

# TODO: Implement fallback to plaintext, for small DataArray (analogous to numpy.ListArrayData)
class DataArray(Serializable, xr.DataArray):
    """
    Stores `xarray.DataArray` objects by converting them to bytes with their
    `.to_netcdf()` method.
    """
    class Data(SerializedData):
        encoding: _EncoderType
        compression: _CompressionType
        summary: str
        data: bytes
        @classmethod
        def encode(cls, dataarray,
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
            v_bytes = dataarray.to_netcdf()
            # Compress and encode the bytes
            array_encoded = encoder(compressor(v_bytes))
            # Return
            return cls(encoding, compression, str(dataarray), array_encoded)
        def decode(data):
            decoder = decoders[data.encoding]
            decompressor = decompressors[data.compression]
            v_bytes = decompressor(decoder(data.data))
            return xr.load_dataarray(v_bytes)   # <<<< Only line which differs from Dataset

# NB: Dataset also provides `.to_netcdf()`, so we can use exactly the same implementation for Dataset.
class Dataset(Serializable, xr.Dataset):
    """
    Stores `xarray.DataArray` objects by converting them to bytes with their
    `.to_netcdf()` method.
    """
    class Data(SerializedData):
        encoding: _EncoderType
        compression: _CompressionType
        summary: str
        data: bytes
        @classmethod
        def encode(cls, dataset,
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
            v_bytes = dataset.to_netcdf()
            # Compress and encode the bytes
            array_encoded = encoder(compressor(v_bytes))
            # Return
            return cls(encoding, compression, str(dataset), array_encoded)
        def decode(data):
            decoder = decoders[data.encoding]
            decompressor = decompressors[data.compression]
            v_bytes = decompressor(decoder(data.data))
            return xr.load_dataset(v_bytes)     # <<<< Only line which differs from DataArray
