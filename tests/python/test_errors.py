import os
import pytest
import numpy as np

from silc import Client, Dataset
from silc.error import RedisReplyError, RedisConnectionError


CLUSTER = True

def test_SSDB_not_set():
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]
    with pytest.raises(RedisConnectionError):
        c = Client(CLUSTER, False)
    os.environ["SSDB"] = ssdb

def test_bad_SSDB():
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]
    os.environ["SSDB"] = "not-an-address:6379;"
    with pytest.raises(RedisConnectionError):
        c = Client(CLUSTER, False)
    os.environ["SSDB"] = ssdb

def test_bad_get_tensor():
    c = Client(CLUSTER, False)
    with pytest.raises(RedisReplyError):
        c.get_tensor("not-a-key")

def test_bad_get_dataset():
    c = Client(CLUSTER, False)
    with pytest.raises(RedisReplyError):
        c.get_dataset("not-a-key")

def test_bad_type_put_dataset():
    c = Client(CLUSTER, False)
    array = np.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        c.put_dataset(array)

def test_bad_type_put_tensor():
    c = Client(CLUSTER, False)
    with pytest.raises(TypeError):
        c.put_tensor("key", [1, 2, 3, 4])

def test_unsupported_type_put_tensor():
    """test an unsupported numpy type
    """
    c = Client(CLUSTER, False)
    data = np.array([1,2,3,4]).astype(np.uint64)
    with pytest.raises(TypeError):
        c.put_tensor("key", data)

def test_bad_type_add_tensor():
    d = Dataset("test-dataset")
    with pytest.raises(TypeError):
        d.add_tensor("test-tensor", [1, 2, 3])

def test_bad_script_file():
    c = Client(True, False)
    with pytest.raises(FileNotFoundError):
        c.set_script_from_file("key", "not-a-file")

def test_bad_callable():
    """user provides none callable function to set_function
    """
    c = Client(True, False)
    with pytest.raises(TypeError):
        c.set_function("key", "not-a-file")

def test_bad_device():
    c = Client(True, False)
    with pytest.raises(TypeError):
        c.set_script("key", "some_script", device="not-a-gpu")

def test_get_non_existant_script():
    c = Client(True, False)
    with pytest.raises(RedisReplyError):
        script = c.get_script("not-a-script")

def test_bad_function_execution():
    """Error raised inside function
    """
    c = Client(True, False)
    c.set_function("bad-function", bad_function)
    data = np.array([1,2,3,4])
    c.put_tensor("bad-func-tensor", data)
    with pytest.raises(RedisReplyError):
        c.run_script("bad-function", "bad_function",
                    ["bad-func-tensor"], ["output"])

def test_missing_script_function():
    """User requests to run a function not in the script"""

    c = Client(True, False)
    c.set_function("bad-function", bad_function)
    with pytest.raises(RedisReplyError):
        c.run_script("bad-function", "not-a-function-in-script",
                    ["bad-func-tensor"], ["output"])

def bad_function(data):
    raise Exception
