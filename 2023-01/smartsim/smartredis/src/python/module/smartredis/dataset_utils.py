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

import functools
import typing as t
from itertools import permutations
from typing import TYPE_CHECKING

from .dataset import Dataset
from .error import RedisRuntimeError
from .util import typecheck

if TYPE_CHECKING:  # pragma: no cover
    # Import optional deps for intellisense
    import xarray as xr

    # Type hint magic bits
    from typing_extensions import ParamSpec

    _PR = ParamSpec("_PR")
    _RT = t.TypeVar("_RT")
else:
    # Leave optional deps as nullish
    xr = None  # pylint: disable=invalid-name

# ----helper decorators -----


def _requires_xarray(fn: "t.Callable[_PR, _RT]") -> "t.Callable[_PR, _RT]":
    @functools.wraps(fn)
    def _import_xarray(*args: "_PR.args", **kwargs: "_PR.kwargs") -> "_RT":
        global xr  # pylint: disable=global-statement,invalid-name
        try:
            import xarray as xr  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise RedisRuntimeError(
                "Optional package xarray must be installed; "
                "Consider running `pip install smartredis[xarray]`"
            ) from e
        return fn(*args, **kwargs)

    return _import_xarray


# ----helper function -----


def get_data(dataset: Dataset, name: str, dtype: str) -> t.List[str]:
    return dataset.get_meta_strings(f"_xarray_{name}_{dtype}_names")[0].split(",")


def typecheck_stringlist(
    names: t.List[str], strings_name: str, string_name: str
) -> None:
    typecheck(names, strings_name, list)
    for name in names:
        typecheck(name, string_name, str)
        # Check if empty
        if string_name == "":
            raise RedisRuntimeError("Empty string")


class DatasetConverter:
    @staticmethod
    def add_metadata_for_xarray(
        dataset: Dataset,
        data_names: t.Union[t.List[str], str],
        dim_names: t.Union[t.List[str], str],
        coord_names: t.Optional[t.Union[t.List[str], str]] = None,
        attr_names: t.Optional[t.Union[t.List[str], str]] = None,
    ) -> None:
        """Extract metadata from a SmartRedis dataset and add it to
        dataformat specific fieldnames

        :param dataset: a Dataset instance
        :type dataset: Dataset
        :param data_names: variable data field name
        :type data_names: list[str] or str
        :param dim_names: dimension field names.
        :type dim_names: list[str]
        :param coord_names: coordinate field names. Defaults to None.
        :type coord_names: list[str], optional
        :param attr_names: attribute field names. Defaults to None.
        :type attr_names: list[str], optional
        """

        if isinstance(data_names, str):
            data_names = [data_names]
        if isinstance(dim_names, str):
            dim_names = [dim_names]
        if isinstance(coord_names, str):
            coord_names = [coord_names]
        if isinstance(attr_names, str):
            attr_names = [attr_names]

        typecheck(dataset, "dataset", Dataset)
        typecheck_stringlist(data_names, "data_names", "data_name")
        typecheck_stringlist(dim_names, "dim_names", "dim_name")
        if coord_names:
            typecheck_stringlist(coord_names, "coord_names", "coord_name")
        if attr_names:
            typecheck_stringlist(attr_names, "attr_names", "attr_name")

        args = [dim_names, coord_names, attr_names]
        sargs = ["dim_names", "coord_names", "attr_names"]

        for name in data_names:
            dataset.add_meta_string("_xarray_data_names", name)
            for arg, sarg in zip(args, sargs):
                if isinstance(arg, list):
                    values = []
                    for val in arg:
                        values.append(val)
                    arg_field = ",".join(values)
                    dataset.add_meta_string(f"_xarray_{name}_{sarg}", arg_field)
                else:
                    if arg:
                        dataset.add_meta_string(f"_xarray_{name}_{sarg}", arg)
                    else:
                        dataset.add_meta_string(f"_xarray_{name}_{sarg}", "null")

    @staticmethod
    @_requires_xarray
    def transform_to_xarray(dataset: Dataset) -> t.Dict:
        """Transform a SmartRedis Dataset, with the appropriate metadata,
        to an Xarray Dataarray

        :param dataset: a Dataset instance
        :type dataset: Dataset

        :return: a dictionary of keys as the data field name and the
        value as the built Xarray DataArray constructed using
        fieldnames and appropriately formatted metadata.
        rtype: dict
        """
        typecheck(dataset, "dataset", Dataset)

        coord_dict = {}
        coord_final = {}
        variable_names = dataset.get_meta_strings("_xarray_data_names")

        # Check for data names that are equal to coordinate names. If any matches
        # are found, then those data variables are treated as coordinates variables
        for tensor_name, tensor_dname in list(permutations(variable_names, 2)):
            for coordname in get_data(dataset, tensor_name, "coord"):
                if tensor_dname == coordname:
                    # Remove coordinate data names from data names
                    if tensor_dname in variable_names:
                        variable_names.remove(tensor_dname)
                    # Get coordinate dimensions in the appropriate format for Xarray
                    coord_dims = []
                    for coord_dim_field_name in get_data(dataset, tensor_dname, "dim"):
                        coord_dims.append(
                            dataset.get_meta_strings(coord_dim_field_name)[0]
                        )
                    # Get coordinate attributes in the appropriate format for Xarray
                    coord_attrs = {}
                    for coord_attr_field_name in get_data(
                        dataset, tensor_dname, "attr"
                    ):
                        fieldname = dataset.get_meta_strings(coord_attr_field_name)[0]
                        coord_attrs[coord_attr_field_name] = fieldname
                    # Add dimensions, data, and attributes to the coordinate variable
                    coord_dict[tensor_dname] = (
                        coord_dims,
                        dataset.get_tensor(tensor_dname),
                        coord_attrs,
                    )
                    # Add coordinate names and relative values in the appropriate
                    # form to add to Xarray coords variable
                    coord_final[tensor_name] = coord_dict

        ret_xarray = {}
        for variable_name in variable_names:
            # Extract dimensions in correct form
            dims_final = [
                dataset.get_meta_strings(dim_field_name)[0]
                for dim_field_name
                in get_data(dataset, variable_name, "dim")
            ]

            # Extract attributes in correct form
            attrs_final = {
                attr_field_name: dataset.get_meta_strings(attr_field_name)[0]
                for attr_field_name
                in get_data(dataset, variable_name, "attr")
            }

            # Add coordinates to the correct data name
            for name, value in coord_final.items():
                if name == variable_name:
                    coords_final = value

            # Construct a xr.DataArray using extracted dataset data,
            # append the dataarray to corresponding variable names
            ret_xarray[variable_name] = xr.DataArray(
                name=variable_name,
                data=dataset.get_tensor(variable_name),
                coords=coords_final,
                dims=dims_final,
                attrs=attrs_final,
            )

        return ret_xarray
