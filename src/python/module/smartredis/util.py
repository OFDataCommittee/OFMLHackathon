# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

from .error import *
import functools
from .smartredisPy import RedisReplyError as PybindRedisReplyError
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

def exception_handler(func):
    """Route exceptions raised in processing SmartRedis API calls to our
    Python wrappers

    :param func: the API function to decorate with this wrapper
    :type func: function
    :raises RedisReplyError: if the wrapped function raised an exception
    """
    @functools.wraps(exception_handler)
    def smartredis_api_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        # Catch RedisReplyErrors for additional processing (convert from
        # pyerror to our error module).
        # TypeErrors and ValueErrors we pass straight through
        except PybindRedisReplyError as cpp_error:
            # query args[0] (i.e. 'self') for the class name
            method_name = args[0].__class__.__name__ + "." + func.__name__
            # Get our exception from the global symbol table.
            # The smartredis.error hierarchy exactly
            # parallels the one built via pybind to enable this
            exception_name = cpp_error.__class__.__name__
            raise globals()[exception_name](str(cpp_error), method_name) from None
    return smartredis_api_wrapper

def typecheck(arg, name, _type):
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
