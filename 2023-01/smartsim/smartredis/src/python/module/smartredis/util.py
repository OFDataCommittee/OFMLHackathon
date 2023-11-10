# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import typing as t
from functools import wraps

import numpy as np
from .error import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .smartredisPy import RedisReplyError as PybindRedisReplyError
from .smartredisPy import c_get_last_error_location

if t.TYPE_CHECKING:
    # Type hint magic bits
    from typing_extensions import ParamSpec

    _PR = ParamSpec("_PR")
    _RT = t.TypeVar("_RT")


class Dtypes:
    @staticmethod
    def tensor_from_numpy(array: np.ndarray) -> str:
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
    def metadata_from_numpy(array: np.ndarray) -> str:
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

    @staticmethod
    def from_string(type_name: str) -> t.Type:
        mapping = {
            "DOUBLE": np.double,
            "FLOAT": np.float64,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "UINT32": np.uint32,
            "UINT64": np.uint64,
            "INT8": np.int8,
            "INT16": np.int16,
            "INT32": np.int32,
            "INT64": np.int64,
            "STRING": str,
        }
        if type_name in mapping:
            return mapping[type_name]
        raise TypeError(f"Unrecognized type name {type_name}")


def init_default(
    default: t.Any, init_value: t.Any, expected_type: t.Optional[t.Any] = None
) -> t.Any:
    """Used for setting a mutable type to a default value.

    PEP standards forbid setting a default value to a mutable type
    Use this function to get around that.
    """
    if init_value is None:
        return default
    if expected_type is not None and not isinstance(init_value, expected_type):
        msg = f"Argument was of type {type(init_value)}, not {expected_type}"
        raise TypeError(msg)
    return init_value


def exception_handler(func: "t.Callable[_PR, _RT]") -> "t.Callable[_PR, _RT]":
    """Route exceptions raised in processing SmartRedis API calls to our
    Python wrappers

    WARNING: using this decorator with a class' @staticmethod or with an
    unbound function that takes a type as its first argument will fail
    because that will make the decorator think it's working with a
    @classmethod

    :param func: the API function to decorate with this wrapper
    :type func: function
    :raises RedisReplyError: if the wrapped function raised an exception
    """

    @wraps(func)
    def smartredis_api_wrapper(*args: "_PR.args", **kwargs: "_PR.kwargs") -> "_RT":
        try:
            return func(*args, **kwargs)
        # Catch RedisReplyErrors for additional processing (convert from
        # pyerror to our error module).
        # TypeErrors and ValueErrors we pass straight through
        except PybindRedisReplyError as cpp_error:
            # get the class for the calling context.
            # for a @classmethod, this will be args[0], but for
            # a "normal" method, args[0] is a self pointer from
            # which we can grab the __class__ attribute
            src_class = args[0]
            if not isinstance(src_class, type):
                src_class = args[0].__class__
            # Build the fully specified name of the calling context
            method_name = src_class.__name__ + "." + func.__name__
            # Get our exception from the global symbol table.
            # The smartredis.error hierarchy exactly
            # parallels the one built via pybind to enable this
            exception_name = cpp_error.__class__.__name__
            error_loc = c_get_last_error_location()
            if error_loc == "unavailable":
                cpp_error_str = str(cpp_error)
            else:
                cpp_error_str = (
                    f"File {error_loc}, in SmartRedis library\n{str(cpp_error)}"
                )
            raise globals()[exception_name](cpp_error_str, method_name) from None

    return smartredis_api_wrapper


def typecheck(arg: t.Any, name: str, _type: t.Union[t.Tuple, type]) -> None:
    """Validate that an argument is of a given type

    :param arg: the variable to be type tested
    :type arg: variable, expected to match _type
    :param name: the name of the variable
    :type name: str
    :param _type: the expected type for the variable
    :type _type: a Python type
    :raises TypeError exception if arg is not of type _type
    """
    if not isinstance(arg, _type):
        raise TypeError(f"Argument {name} is of type {type(arg)}, not {_type}")
