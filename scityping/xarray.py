import xarray as xr
from typing import Union, Tuple

from .base import Serializable
from .base_types import SerializedData
from .numpy import (encoders, decoders, compressors, decompressors,
                    _EncoderType, _CompressionType)

# NB: Encoding closely emulates that for numpy.Array
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
            return xr.load_dataarray(v_bytes)

# TODO: Dataset