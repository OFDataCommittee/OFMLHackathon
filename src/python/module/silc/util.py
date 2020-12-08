
class Dtypes:

    @staticmethod
    def tensor_from_numpy(array):
        mapping = {
            "float64": "DOUBLE",
            "float32": "FLOAT",
            "uint8": "UINT8",
            "uint16": "UINT16",
            "int8": "INT8",
            "int16": "INT16",
            "int32": "INT32",
            "int64": "INT64"
        }
        dtype = str(array.dtype)
        if dtype in mapping:
            return mapping[dtype]
        else:
            raise TypeError(f"Incompatible tensor type provided {dtype}")

