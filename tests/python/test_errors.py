import pytest
import numpy as np

from silc import Client, Dataset
from silc.error import RedisReplyError

CLUSTER = True

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
    array = np.array([1,2,3,4])
    with pytest.raises(TypeError):
        c.put_dataset(array)

def test_bad_type_put_tensor():
    c = Client(CLUSTER, False)
    with pytest.raises(TypeError):
        c.put_tensor([1,2,3,4])

def test_bad_type_add_tensor():
    d = Dataset("test-dataset")
    with pytest.raises(TypeError):
        d.add_tensor("test-tensor", [1,2,3])
