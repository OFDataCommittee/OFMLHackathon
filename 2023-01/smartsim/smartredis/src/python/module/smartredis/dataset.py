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

import numpy as np

from .smartredisPy import PyDataset
from .srobject import SRObject
from .util import Dtypes, exception_handler, typecheck


class Dataset(SRObject):
    def __init__(self, name: str) -> None:
        """Initialize a Dataset object

        :param name: name of dataset
        :type name: str
        """
        super().__init__(PyDataset(name))
        typecheck(name, "name", str)

    def __str__(self) -> str:
        """Create a string representation of the client

        :return: A string representation of the client
        :rtype: str
        """
        return self._data.to_string()

    @property
    def _data(self) -> PyDataset:
        """Alias _srobject to _data"""
        assert isinstance(self._srobject, PyDataset)
        return self._srobject

    @staticmethod
    def from_pybind(dataset: PyDataset) -> "Dataset":
        """Initialize a Dataset object from a PyDataset object

            Create a new Dataset object using the data and properties 
            of a PyDataset object as the initial values.

        :param dataset: The pybind PyDataset object
                        to use for construction
        :type dataset: PyDataset
        :return: The newly constructed Dataset object
        :rtype: Dataset object
        """
        typecheck(dataset, "dataset", PyDataset)
        new_dataset = Dataset(dataset.get_name())
        new_dataset.set_data(dataset)
        return new_dataset

    @exception_handler
    def get_data(self) -> PyDataset:
        """Return the PyDataset attribute

        :return: The PyDataset attribute containing
                 the dataset information
        :rtype: PyDataset
        """
        return self._data

    @exception_handler
    def set_data(self, dataset: PyDataset) -> None:
        """Set the PyDataset attribute

        :param dataset: The PyDataset object
        :type dataset: PyDataset
        """
        typecheck(dataset, "dataset", PyDataset)
        self._srobject = dataset

    @exception_handler
    def add_tensor(self, name: str, data: np.ndarray) -> None:
        """Add a named multi-dimensional data array (tensor) to this dataset

        :param name: name associated to the tensor data
        :type name: str
        :param data: tensor data
        :type data: np.ndarray
        """
        typecheck(name, "name", str)
        typecheck(data, "data", np.ndarray)
        dtype = Dtypes.tensor_from_numpy(data)
        self._data.add_tensor(name, data, dtype)

    @exception_handler
    def get_tensor(self, name: str) -> np.ndarray:
        """Get a tensor from the Dataset

        :param name: name of the tensor to get
        :type name: str
        :return: a numpy array of tensor data
        :rtype: np.ndarray
        """
        typecheck(name, "name", str)
        return self._data.get_tensor(name)

    @exception_handler
    def get_name(self) -> str:
        """Get the name of a Dataset

        :return: the name of the in-memory dataset
        :rtype: str
        """
        return self._data.get_name()

    @exception_handler
    def add_meta_scalar(self, name: str, data: t.Union[int, float]) -> None:
        """Add scalar (non-string) metadata to a field name if it exists; 
        otherwise, create and add

            If the field name exists, append the scalar metadata; otherwise, 
            create the field within the DataSet object and add the scalar metadata.

        :param name: The name used to reference the scalar metadata field
        :type name: str
        :param data: scalar metadata input
        :type data: int | float
        """
        typecheck(name, "name", str)

        # We want to support numpy datatypes and avoid pybind ones
        data_as_array = np.asarray(data)
        if data_as_array.size > 1:
            raise TypeError("Argument provided is not a scalar")
        # We keep dtype, in case data has a non-standard python format
        dtype = Dtypes.metadata_from_numpy(data_as_array)
        self._data.add_meta_scalar(name, data_as_array, dtype)

    @exception_handler
    def add_meta_string(self, name: str, data: str) -> None:
        """Add string metadata to a field name if it exists; otherwise, create and add

            If the field name exists, append the string metadata; otherwise, 
            create the field within the DataSet object and add the string metadata.

        :param name: The name used to reference the string metadata field
        :type name: str
        :param data: string metadata input
        :type data: str
        """
        typecheck(name, "name", str)
        typecheck(data, "data", str)
        self._data.add_meta_string(name, data)

    @exception_handler
    def get_meta_scalars(self, name: str) -> t.Union[t.List[int], t.List[float]]:
        """Get the scalar values from the DataSet assigned to a field name

        :param name: The field name to retrieve from
        :type name: str
        :rtype: list[int] | list[float]
        """
        typecheck(name, "name", str)
        return self._data.get_meta_scalars(name)

    @exception_handler
    def get_meta_strings(self, name: str) -> t.List[str]:
        """Get the string values from the DataSet assigned to a field name

        :param name: The field name to retrieve from
        :type name: str
        :rtype: list[str]
        """
        typecheck(name, "name", str)
        return self._data.get_meta_strings(name)

    @exception_handler
    def get_metadata_field_names(self) -> t.List[str]:
        """Get all field names from the DataSet

        :return: a list of all metadata field names
        :rtype: list[str]
        """
        return self._data.get_metadata_field_names()

    @exception_handler
    def get_metadata_field_type(self, name: str) -> t.Type:
        """Get the type of metadata for a field name (scalar or string)

        :param name: The name used to reference the metadata
                     field in the DataSet
        :type name: str
        :return: the numpy type for the metadata field
        :rtype: type
        """
        typecheck(name, "name", str)
        type_str = self._data.get_metadata_field_type(name)
        return Dtypes.from_string(type_str)

    @exception_handler
    def get_tensor_type(self, name: str) -> t.Type:
        """Get the type of a tensor in the DataSet

        :param name: The name used to reference the tensor in the DataSet
        :type name: str
        :return: the numpy type for the tensor
        :rtype: type
        """
        typecheck(name, "name", str)
        type_str = self._data.get_tensor_type(name)
        return Dtypes.from_string(type_str)

    @exception_handler
    def get_tensor_names(self) -> t.List[str]:
        """Get the names of all tensors in the DataSet

        :return: a list of tensor names
        :rtype: list[str]
        """
        return self._data.get_tensor_names()

    @exception_handler
    def get_tensor_dims(self, name: str) -> t.List[int]:
        """Get the dimensions of a tensor in the DataSet

        :param name: name associated to the tensor data
        :type name: str
        :return: a list of the tensor dimensions
        :rtype: list[int]
        """
        typecheck(name, "name", str)
        return self._data.get_tensor_dims(name)
