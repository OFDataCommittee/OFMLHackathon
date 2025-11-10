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
from uuid import uuid4

from .error import RedisRuntimeError
from .smartredisPy import PyConfigOptions
from .util import exception_handler, typecheck


class _Managed:
    """Marker class identifying factory-created objects"""


def create_managed_instance(base: t.Type[t.Any]) -> t.Any:
    """Instantiate a managed instance of the class, enabling the use of type 
    checking to detect if an instance is managed"""
    def get_dynamic_class_name(bases: t.Tuple[t.Type]) -> str:
        """Create a name for the new type by concatenating base names. Appends a
        unique suffix to avoid confusion if dynamic type comparisons occur"""
        unique_key = str(uuid4()).split("-", 1)[0]
        class_name = "".join(base.__name__ for base in bases) + unique_key
        return class_name

    # Create a subtype that includes the _Managed marker
    bases = (_Managed, base)
    class_name = get_dynamic_class_name(bases)
    managed_class = type(class_name, bases, {})
    return managed_class()


def managed(func: t.Callable) -> t.Callable:
    """Decorator to verify that a class was constructed using a factory"""
    not_managed = (
        "Attempting to call managed method on ConfigOptions object not "
        "created from a factory method"
    )

    @wraps(func)
    def _wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        instance = args[0]
        if not isinstance(instance, _Managed):
            msg = not_managed.format(instance.__class__.__name__)
            raise RedisRuntimeError(msg)
        return func(*args, **kwargs)

    return _wrapper


class ConfigOptions:
    def __init__(self) -> None:
        """Initialize a ConfigOptions base object"""
        self._config_opts: t.Any = None

    @staticmethod
    def from_pybind(configoptions: PyConfigOptions) -> "ConfigOptions":
        """Initialize a ConfigOptions object from a PyConfigOptions object

        :param configoptions: The pybind PyConfigOptions object
                              to use for construction
        :type dataset: PyConfigOptions
        :return: The newly constructed ConfigOptions from the PyConfigOptions
        :rtype: ConfigOptions
        """
        typecheck(configoptions, "configoptions", PyConfigOptions)
        opts: ConfigOptions = create_managed_instance(ConfigOptions)
        opts.set_configoptions(configoptions)
        return opts

    @exception_handler
    @managed
    def get_data(self):
        """Return the PyConfigOptions attribute

        :return: The PyConfigOptions attribute containing
                 the ConfigOptions information
        :rtype: PyConfigOptions
        """
        return self._config_opts

    @exception_handler
    def set_configoptions(self, configoptions: PyConfigOptions) -> None:
        """Set the PyConfigOptions attribute

        :param configoptions: The PyConfigOptions object
        :type configoptions: PyConfigOptions
        """
        typecheck(configoptions, "configoptions", PyConfigOptions)
        self._config_opts = configoptions

    @classmethod
    @exception_handler
    def create_from_environment(cls, db_suffix: str) -> "ConfigOptions":
        """Instantiate ConfigOptions, getting selections from
        environment variables. If db_suffix is non-empty,
        then "_{db_suffix}" will be appended to the name of
        each environment variable that is read

        :param cls: The ConfigOptions class
        :type cls: type
        :param db_suffix: Prefix to append to environment variables
                          or an empty string to eschew appending
        :type db_suffix: str
        :return: An instantiated ConfigOptions object
        :rtype: ConfigOptions
        """
        typecheck(db_suffix, "db_suffix", str)
        configoptions = PyConfigOptions.create_from_environment(db_suffix)
        opts: ConfigOptions = create_managed_instance(ConfigOptions)
        opts.set_configoptions(configoptions)
        return opts

    @exception_handler
    @managed
    def get_integer_option(self, option_name: str) -> int:
        """Retrieve the value of a numeric configuration option
        from the selected source

        :param option_name: The name of the configuration option to retrieve
        :type option_name: str
        :return: The value of the selected option. Returns
                 default_value if the option was not set in the
                 selected source
        :rtype: int
        """
        typecheck(option_name, "option_name", str)
        return self._config_opts.get_integer_option(option_name)

    @exception_handler
    @managed
    def get_string_option(self, option_name: str) -> str:
        """Retrieve the value of a string configuration option
        from the selected source

        :param option_name: The name of the configuration option to retrieve
        :type option_name: str
        :return: The value of the selected option. Returns
                 default_value if the option was not set in the
                 selected source
        :rtype: str
        """
        typecheck(option_name, "option_name", str)
        return self._config_opts.get_string_option(option_name)

    @exception_handler
    @managed
    def is_configured(self, option_name: str) -> bool:
        """Check whether a configuration option is set in the selected source

        :param option_name: The name of the configuration option to check
        :type option_name: str
        :return: True IFF the target option is defined in the selected source
                 or if it has been overridden
        :rtype: bool
        """
        typecheck(option_name, "option_name", str)
        return self._config_opts.is_configured(option_name)

    @exception_handler
    @managed
    def override_integer_option(self, option_name: str, value: int) -> None:
        """Override the value of a numeric configuration option
        in the selected source

        Overrides are specific to an instance of the
        ConfigOptions class. An instance that references
        the same source will not be affected by an override to
        a different ConfigOptions instance

        :param option_name: The name of the configuration option to override
        :type option_name: str
        :param value: The value to store for the configuration option
        :type value: int
        """
        typecheck(option_name, "option_name", str)
        typecheck(value, "value", int)
        self._config_opts.override_integer_option(option_name, value)

    @exception_handler
    @managed
    def override_string_option(self, option_name: str, value: str) -> None:
        """Override the value of a string configuration option
        in the selected source

        Overrides are specific to an instance of the
        ConfigOptions class. An instance that references
        the same source will not be affected by an override to
        a different ConfigOptions instance

        :param option_name: The name of the configuration option to override
        :type option_name: str
        :param value: The value to store for the configuration option
        :type value: str
        """
        typecheck(option_name, "option_name", str)
        typecheck(value, "value", str)
        self._config_opts.override_string_option(option_name, value)
