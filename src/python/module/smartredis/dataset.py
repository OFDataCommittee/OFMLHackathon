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

from numbers import Number

import numpy as np

from .smartredisPy import PyDataset
from .util import Dtypes, exception_handler, typecheck

from .error import *

class Dataset:
    def __init__(self, name):
        """Initialize a Dataset object

        :param name: name of dataset
        :type name: str
        """
        typecheck(name, "name", str)
        self._data = PyDataset(name)

    @staticmethod
    def from_pybind(dataset):
        """Initialize a Dataset object from
        a PyDataset object

        :param dataset: The pybind PyDataset object
                        to use for construction
        :type dataset: PyDataset
        :return: The newly constructor Dataset from
                 the PyDataset
        :rtype: Dataset
        """
        typecheck(dataset, "dataset", PyDataset)
        new_dataset = Dataset(dataset.get_name())
        new_dataset.set_data(dataset)
        return new_dataset

    @exception_handler
    def get_data(self):
        """Return the PyDataset attribute

        :return: The PyDataset attribute containing
                 the dataset information
        :rtype: PyDataset
        """
        return self._data

    @exception_handler
    def set_data(self, dataset):
        """Set the PyDataset attribute

        :param dataset: The PyDataset object
        :type dataset: PyDataset
        """
        typecheck(dataset, "dataset", PyDataset)
        self._data = dataset

    @exception_handler
    def add_tensor(self, name, data):
        """Add a named tensor to this dataset

        :param name: tensor name
        :type name: str
        :param data: tensor data
        :type data: np.array
        """
        typecheck(name, "name", str)
        typecheck(data, "data", np.ndarray)
        dtype = Dtypes.tensor_from_numpy(data)
        self._data.add_tensor(name, data, dtype)

    @exception_handler
    def get_tensor(self, name):
        """Get a tensor from the Dataset

        :param name: name of the tensor to get
        :type name: str
        :return: a numpy array of tensor data
        :rtype: np.array
        """
        typecheck(name, "name", str)
        return self._data.get_tensor(name)

    @exception_handler
    def add_meta_scalar(self, name, data):
        """Add metadata scalar field (non-string) with value to the DataSet

            If the field does not exist, it will be created.
            If the field exists, the value
            will be appended to existing field.

        :param name: The name used to reference the metadata
                     field
        :type name: str
        :param data: a scalar
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
    def add_meta_string(self, name, data):
        """Add metadata string field with value to the DataSet

        If the field does not exist, it will be created
        If the field exists the value will
        be appended to existing field.

        :param name: The name used to reference the metadata
                     field
        :type name: str
        :param data: The string to add to the field
        :type data: str
        """
        typecheck(name, "name", str)
        typecheck(data, "data", str)
        self._data.add_meta_string(name, data)

    @exception_handler
    def get_meta_scalars(self, name):
        """Get the metadata scalar field values from the DataSet

        :param name: The name used to reference the metadata
                     field in the DataSet
        :type name: str
        """
        typecheck(name, "name", str)
        return self._data.get_meta_scalars(name)

    @exception_handler
    def get_meta_strings(self, name):
        """Get the metadata scalar field values from the DataSet

        :param name: The name used to reference the metadata
                        field in the DataSet
        :type name: str
        """
        typecheck(name, "name", str)
        return self._data.get_meta_strings(name)
