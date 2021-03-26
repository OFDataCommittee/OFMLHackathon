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
            "int64": "INT64",
        }
        dtype = str(array.dtype)
        if dtype in mapping:
            return mapping[dtype]
        raise TypeError(f"Incompatible tensor type provided {dtype}")

    @staticmethod
    def metadata_from_numpy(array):
        mapping = {
            "float64": "DOUBLE",
            "float32": "FLOAT",
            "uint32": "UINT32",
            "uint64": "UINT64",
            "int32": "INT32",
            "int64": "INT64",
        }
        dtype = str(array.dtype)
        if dtype in mapping:
            return mapping[dtype]
        raise TypeError(f"Incompatible metadata type provided {dtype}")


def init_default(default, init_value, expected_type=None):
    """Used for setting a mutable type to a default value.

    PEP standards forbid setting a default value to a mutable type
    Use this function to get around that.
    """
    if init_value is None:
        return default
    if expected_type is not None and not isinstance(init_value, expected_type):
        raise TypeError(f"Argument was of type {type(init_value)}, not {expected_type}")
    return init_value
