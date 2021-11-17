# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

import time

import numpy as np
from smartredis import Client, Dataset


def test_put_get_dataset(mock_data, use_cluster):
    """test sending and recieving a dataset with 2D tensors
    of every datatype
    """

    data = mock_data.create_data((10, 10))

    # Create a dataset to put
    dataset = Dataset("test-dataset")
    for index, tensor in enumerate(data):
        key = f"tensor_{str(index)}"
        dataset.add_tensor(key, tensor)

    client = Client(None, use_cluster)

    assert not client.dataset_exists(
        "nonexistent-dataset"
    ), "Existence of nonexistant dataset!"

    client.put_dataset(dataset)

    assert client.dataset_exists("test-dataset"), "Non-existance of real dataset!"

    rdataset = client.get_dataset("test-dataset")
    for index, tensor in enumerate(data):
        key = f"tensor_{str(index)}"
        rtensor = rdataset.get_tensor(key)
        np.testing.assert_array_equal(
            rtensor,
            tensor,
            "Dataset returned from get_dataset not the same as sent dataset",
        )


def test_augment_dataset(mock_data, use_cluster):
    """Test sending, receiving, altering, and sending
    a Dataset.
    """
    # Create mock test data
    data = mock_data.create_data((10, 10))

    # Set test dataset name
    dataset_name = "augment-dataset"

    # Initialize a client
    client = Client(None, use_cluster)

    # Create a dataset to put into the database
    dataset = Dataset(dataset_name)
    for index, tensor in enumerate(data):
        tensor_name = f"tensor_{str(index)}"
        dataset.add_tensor(tensor_name, tensor)

    # Send the dataset
    client.put_dataset(dataset)

    # Retrieve the dataset
    rdataset = client.get_dataset(dataset_name)

    # Add a new tensor to the retrieved dataset
    new_tensor = np.array([1.0, 2.0, 3.0])
    rdataset.add_tensor("new_tensor", new_tensor)

    # Add a metadata scalar field to the dataset
    scalar_field = 1.0
    scalar_name = "scalar_field_1"
    rdataset.add_meta_scalar(scalar_name, scalar_field)

    # Add a metadata string field to the dataset
    string_field = "test_string"
    string_name = "string_field"
    rdataset.add_meta_string(string_name, string_field)

    # Send the augmented dataset
    client.put_dataset(rdataset)

    # Retrieve the augmented dataset
    aug_dataset = client.get_dataset(dataset_name)

    # Check the accuracy of the augmented dataset
    for index, tensor in enumerate(data):
        tensor_name = f"tensor_{str(index)}"
        rtensor = aug_dataset.get_tensor(tensor_name)
        np.testing.assert_array_equal(
            rtensor,
            tensor,
            "Dataset returned from get_dataset not the same as sent dataset",
        )

    rtensor = aug_dataset.get_tensor("new_tensor")
    np.testing.assert_array_equal(
        rtensor,
        new_tensor,
        "Dataset returned did not return the correct additional tensor",
    )

    # Check the accuracy of the metadat fields
    assert aug_dataset.get_meta_scalars(scalar_name).size == 1
    assert len(aug_dataset.get_meta_strings(string_name)) == 1
    assert aug_dataset.get_meta_scalars(scalar_name)[0] == scalar_field
    assert aug_dataset.get_meta_strings(string_name)[0] == string_field
