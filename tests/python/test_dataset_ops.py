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

    assert not (client.key_exists(get_prefix() + "dataset_rename"))
    assert client.key_exists(get_prefix() + "dataset_renamed")
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
