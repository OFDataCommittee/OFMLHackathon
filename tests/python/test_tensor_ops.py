import numpy as np
import pytest

from smartredis import Client
from smartredis.error import RedisReplyError


def test_copy_tensor(use_cluster):
    # test copying tensor

    client = Client(None, use_cluster)
    tensor = np.array([1, 2])
    client.put_tensor("test_copy", tensor)

    client.copy_tensor("test_copy", "test_copied")

    assert client.key_exists("test_copy")
    assert client.key_exists("test_copied")
    returned = client.get_tensor("test_copied")
    assert np.array_equal(tensor, returned)


def test_rename_tensor(use_cluster):
    # test renaming tensor

    client = Client(None, use_cluster)
    tensor = np.array([1, 2])
    client.put_tensor("test_rename", tensor)

    client.rename_tensor("test_rename", "test_renamed")

    assert not (client.key_exists("test_rename"))
    assert client.key_exists("test_renamed")
    returned = client.get_tensor("test_renamed")
    assert np.array_equal(tensor, returned)


def test_delete_tensor(use_cluster):
    # test renaming tensor

    client = Client(None, use_cluster)
    tensor = np.array([1, 2])
    client.put_tensor("test_delete", tensor)

    client.delete_tensor("test_delete")

    assert not (client.key_exists("test_delete"))


# --------------- Error handling ----------------------


def test_rename_nonexisting_key(use_cluster):

    client = Client(None, use_cluster)
    with pytest.raises(RedisReplyError):
        client.rename_tensor("not-a-tensor", "still-not-a-tensor")


def test_copy_nonexistant_key(use_cluster):

    client = Client(None, use_cluster)
    with pytest.raises(RedisReplyError):
        client.copy_tensor("not-a-tensor", "still-not-a-tensor")


def test_copy_not_tensor(use_cluster):
    def test_func(param):
        print(param)

    client = Client(None, use_cluster)
    client.set_function("test_func", test_func)
    with pytest.raises(RedisReplyError):
        client.copy_tensor("test_func", "test_fork")
