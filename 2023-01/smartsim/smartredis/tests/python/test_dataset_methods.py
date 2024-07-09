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

import numpy as np
from smartredis import Dataset

def test_serialize_dataset():
    """Test serializing a dataset
    """
    dataset = Dataset("test-dataset")
    data = np.uint8([2,4,8])
    dataset.add_tensor("u8_tensor", data)
    data = np.double([2.0,4.1,8.3, 5.6])
    dataset.add_tensor("double_tensor", data)
    dataset.add_meta_scalar("float2_scalar", float(3.1415926535))
    dataset.add_meta_scalar("float_scalar", np.double(3.1415926535))
    dataset.add_meta_string("metastring", "metavalue")
    assert str(dataset) != repr(dataset)


def test_add_get_tensor(mock_data):
    """Test adding and retrieving 1D tensors to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 1D tensors of all data types
    data = mock_data.create_data(10)
    add_get_arrays(dataset, data)


def test_add_get_tensor_2D(mock_data):
    """Test adding and retrieving 2D tensors to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 2D tensors of all data types
    data_2D = mock_data.create_data((10, 10))
    add_get_arrays(dataset, data_2D)


def test_add_get_tensor_3D(mock_data):
    """Test adding and retrieving 3D tensors to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 3D tensors of all datatypes
    data_3D = mock_data.create_data((10, 10, 10))
    add_get_arrays(dataset, data_3D)


def test_add_get_scalar(mock_data):
    """Test adding and retrieving scalars to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 1D tensors of all data types
    data = mock_data.create_metadata_scalars(10)
    add_get_scalars(dataset, data)


def test_add_get_strings(mock_data):
    """Test adding and retrieving strings to
    a dataset
    """
    dataset = Dataset("test-dataset")

    # list of strings
    data = mock_data.create_metadata_strings(10)
    add_get_strings(dataset, data)


def test_dataset_inspection(context):
    d = Dataset(context)
    data = np.uint8([[2, 4, 6, 8], [1, 3, 5, 7]])
    d.add_tensor("u8_tensor", data)
    data = np.int16([1, 1, 2, 3, 5, 8])
    d.add_tensor("i16_tensor", data)
    d.add_meta_string("metastring", "metavalue")
    d.add_meta_scalar("u32_scalar", np.uint32(42))
    d.add_meta_scalar("double_scalar", np.double(3.1415926535))
    dims = d.get_tensor_dims("u8_tensor")
    assert len(dims) == 2
    assert dims[0] == 2
    assert dims[1] == 4

    tensornames = d.get_tensor_names()
    assert 2 == len(tensornames)
    tensornames.sort()
    assert "i16_tensor" == tensornames[0]
    assert "u8_tensor" == tensornames[1]
    assert np.uint8 == d.get_tensor_type("u8_tensor")
    assert np.int16 == d.get_tensor_type("i16_tensor")

    metanames = d.get_metadata_field_names()
    assert 3 == len(metanames)
    metanames.sort()
    assert "double_scalar" == metanames[0]
    assert "metastring" == metanames[1]
    assert "u32_scalar" == metanames[2]
    assert np.float64 == d.get_metadata_field_type("double_scalar")
    assert str == d.get_metadata_field_type("metastring")
    assert np.uint32 == d.get_metadata_field_type("u32_scalar")

# ------- Helper Functions -----------------------------------------------


def add_get_arrays(dataset, data):
    """Helper for dataset tests"""

    # add to dataset
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        dataset.add_tensor(key, array)

    # get from dataset
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        rarray = dataset.get_tensor(key)
        np.testing.assert_array_equal(
            rarray,
            array,
            "Returned array from get_tensor not equal to tensor added to dataset",
        )


def add_get_scalars(dataset, data):
    """Helper for metadata tests"""

    # add to dataset
    for index, scalars in enumerate(data):
        key = f"meta_scalars_{index}"
        for scalar in scalars:
            dataset.add_meta_scalar(key, scalar)

    # get from dataset
    for index, scalars in enumerate(data):
        key = f"meta_scalars_{index}"
        rscalars = dataset.get_meta_scalars(key)
        np.testing.assert_array_equal(
            rscalars,
            scalars,
            "Returned scalars from get_meta_scalars not equal to scalars added to dataset",
        )


def add_get_strings(dataset, data):
    """Helper for metadata tests"""

    # add to dataset
    key = "test_meta_strings"
    for meta_string in data:
        dataset.add_meta_string(key, meta_string)

    # get from dataset
    rdata = dataset.get_meta_strings(key)
    assert len(data) == len(rdata)
    assert all([a == b for a, b in zip(data, rdata)])
