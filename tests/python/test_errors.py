import os

import numpy as np
import pytest

from silc import Client, Dataset
from silc.error import RedisConnectionError, RedisReplyError


def test_SSDB_not_set(use_cluster):
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]
    with pytest.raises(RedisConnectionError):
        c = Client(None, use_cluster)
    os.environ["SSDB"] = ssdb


def test_bad_SSDB(use_cluster):
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]
    os.environ["SSDB"] = "not-an-address:6379;"
    with pytest.raises(RedisConnectionError):
        c = Client(None, use_cluster)
    os.environ["SSDB"] = ssdb


def test_bad_get_tensor(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(RedisReplyError):
        c.get_tensor("not-a-key")


def test_bad_get_dataset(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(RedisReplyError):
        c.get_dataset("not-a-key")


def test_bad_type_put_dataset(use_cluster):
    c = Client(None, use_cluster)
    array = np.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        c.put_dataset(array)


def test_bad_type_put_tensor(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.put_tensor("key", [1, 2, 3, 4])


def test_unsupported_type_put_tensor(use_cluster):
    """test an unsupported numpy type"""
    c = Client(None, use_cluster)
    data = np.array([1, 2, 3, 4]).astype(np.uint64)
    with pytest.raises(TypeError):
        c.put_tensor("key", data)


def test_bad_type_add_tensor(use_cluster):
    d = Dataset("test-dataset")
    with pytest.raises(TypeError):
        d.add_tensor("test-tensor", [1, 2, 3])


def test_bad_script_file(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(FileNotFoundError):
        c.set_script_from_file("key", "not-a-file")


def test_bad_callable(use_cluster):
    """user provides none callable function to set_function"""
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_function("key", "not-a-file")


def test_bad_device(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_script("key", "some_script", device="not-a-gpu")


def test_get_non_existant_script(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(RedisReplyError):
        script = c.get_script("not-a-script")


def test_bad_function_execution(use_cluster):
    """Error raised inside function"""

    c = Client(None, use_cluster)
    c.set_function("bad-function", bad_function)
    data = np.array([1, 2, 3, 4])
    c.put_tensor("bad-func-tensor", data)
    with pytest.raises(RedisReplyError):
        c.run_script("bad-function", "bad_function", ["bad-func-tensor"], ["output"])


def test_missing_script_function(use_cluster):
    """User requests to run a function not in the script"""

    c = Client(None, use_cluster)
    c.set_function("bad-function", bad_function)
    with pytest.raises(RedisReplyError):
        c.run_script("bad-function", "not-a-function-in-script", ["bad-func-tensor"], ["output"])


def bad_function(data):
    """Bad function which only raises an exception"""
    raise Exception
