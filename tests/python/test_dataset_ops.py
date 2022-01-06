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

import os
import numpy as np
import pytest
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError


def test_copy_dataset(use_cluster):
    # test copying dataset from one key to another

    dataset = create_dataset("test_dataset_copy")

    client = Client(None, use_cluster)
    client.put_dataset(dataset)

    client.copy_dataset("test_dataset_copy", "test_dataset_copied")

    # ensure copied dataset is the same after deleting the orig dataset
    client.delete_dataset("test_dataset_copy")
    returned = client.get_dataset("test_dataset_copied")

    # assert copied array is the same
    array = dataset.get_tensor("test_array")
    copied_array = returned.get_tensor("test_array")
    np.testing.assert_array_equal(
        array, copied_array, "arrays in copied dataset are not equal"
    )

    # assert copied meta string is the same
    string = dataset.get_meta_strings("test_string")
    copied_string = returned.get_meta_strings("test_string")
    np.testing.assert_array_equal(
        string, copied_string, "strings in copied dataset are not equal"
    )

    # assert copied meta scaler is the same
    scalar = dataset.get_meta_scalars("test_scalar")
    copied_scalar = returned.get_meta_scalars("test_scalar")
    np.testing.assert_array_equal(
        scalar, copied_scalar, "scalars in copied dataset are not equal"
    )


def test_rename_dataset(use_cluster):
    # test renaming a dataset in the database

    dataset = create_dataset("dataset_rename")

    client = Client(None, use_cluster)
    client.put_dataset(dataset)

    client.rename_dataset("dataset_rename", "dataset_renamed")

    assert not (client.dataset_exists("dataset_rename"))
    assert not (client.poll_dataset("dataset_rename", 50, 5))
    assert client.dataset_exists("dataset_renamed")
    assert client.poll_dataset("dataset_renamed", 50, 5)
    returned = client.get_dataset("dataset_renamed")

    # assert copied array is the same
    array = dataset.get_tensor("test_array")
    copied_array = returned.get_tensor("test_array")
    np.testing.assert_array_equal(
        array, copied_array, "arrays in renamed dataset are not equal"
    )

    # assert copied meta string is the same
    string = dataset.get_meta_strings("test_string")
    copied_string = returned.get_meta_strings("test_string")
    np.testing.assert_array_equal(
        string, copied_string, "strings in renamed dataset are not equal"
    )

    # assert copied meta scalar is the same
    scalar = dataset.get_meta_scalars("test_scalar")
    copied_scalar = returned.get_meta_scalars("test_scalar")
    np.testing.assert_array_equal(
        scalar, copied_scalar, "scalars in renamed dataset are not equal"
    )


def test_delete_dataset(use_cluster):
    # test renaming a dataset in the database

    dataset = create_dataset("dataset_delete")

    client = Client(None, use_cluster)
    client.put_dataset(dataset)

    client.delete_dataset(
        "dataset_delete",
    )

    assert not (client.key_exists("dataset_delete"))


# ----------- Error handling ------------------------------------


def test_rename_nonexisting_dataset(use_cluster):

    client = Client(None, use_cluster)
    with pytest.raises(RedisReplyError):
        client.rename_dataset("not-a-tensor", "still-not-a-tensor")


def test_copy_nonexistant_dataset(use_cluster):

    client = Client(None, use_cluster)
    with pytest.raises(RedisReplyError):
        client.copy_dataset("not-a-tensor", "still-not-a-tensor")


def test_copy_not_dataset(use_cluster):
    def test_func(param):
        print(param)

    client = Client(None, use_cluster)
    client.set_function("test_func_dataset", test_func)
    with pytest.raises(RedisReplyError):
        client.copy_dataset("test_func_dataset", "test_fork_dataset")


# ------------ helper functions ---------------------------------


def create_dataset(name):
    array = np.array([1, 2, 3, 4])
    string = "test_meta_strings"
    scalar = 7

    dataset = Dataset(name)
    dataset.add_tensor("test_array", array)
    dataset.add_meta_string("test_string", string)
    dataset.add_meta_scalar("test_scalar", scalar)
    return dataset


def get_prefix():
    # get prefix, if it exists. Assumes the client is using
    # tensor prefix which is the default.
    sskeyin = os.environ.get("SSKEYIN", None)
    prefix = ""
    if sskeyin:
        prefix = sskeyin.split(",")[0] + "."
    return prefix
