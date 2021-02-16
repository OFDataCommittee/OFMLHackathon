import inspect
import os.path as osp

import numpy as np
import torch

from silc import Client
import get_cluster_env

CLUSTER = get_cluster_env.cluster()
file_path = osp.dirname(osp.abspath(__file__))


def test_set_get_function():
    c = Client(None, CLUSTER)
    c.set_function("test-set-function", one_to_one)
    script = c.get_script("test-set-function")
    sent_script = inspect.getsource(one_to_one)
    assert script == sent_script


def test_set_get_script():
    c = Client(None, CLUSTER)
    sent_script = read_script_from_file()
    c.set_script("test-set-script", sent_script)
    script = c.get_script("test-set-script")
    assert sent_script == script


def test_set_script_from_file():
    sent_script = read_script_from_file()
    c = Client(None, CLUSTER)
    c.set_script_from_file(
        "test-script-file", osp.join(file_path, "./data_processing_script.txt")
    )
    returned_script = c.get_script("test-script-file")
    assert sent_script == returned_script


def test_run_script():
    data = np.array([[1, 2, 3, 4, 5]])

    c = Client(None, CLUSTER)
    c.put_tensor("script-test-data", data)
    c.set_function("one-to-one", one_to_one)
    c.run_script("one-to-one", "one_to_one", ["script-test-data"], ["script-test-out"])
    out = c.get_tensor("script-test-out")
    assert out == 5


def test_run_script_multi():
    data = np.array([[1, 2, 3, 4]])
    data_2 = np.array([[5, 6, 7, 8]])

    c = Client(None, CLUSTER)
    c.put_tensor("srpt-multi-out-data-1", data)
    c.put_tensor("srpt-multi-out-data-2", data_2)
    c.set_function("two-to-one", two_to_one)
    c.run_script(
        "two-to-one",
        "two_to_one",
        ["srpt-multi-out-data-1", "srpt-multi-out-data-2"],
        ["srpt-multi-out-output"],
    )
    out = c.get_tensor("srpt-multi-out-output")
    expected = np.array([4, 8])
    np.testing.assert_array_equal(
        out, expected, "Returned array from script not equal to expected result"
    )


def one_to_one(data):
    """Sample torchscript script that returns the
    highest element in an array.

    One input to one output
    """
    # return the highest element
    return data.max(1)[0]


def two_to_one(data, data_2):
    """Sample torchscript script that returns the
    highest elements in both arguments

    One input to one output
    """
    # return the highest element
    merged = torch.cat((data, data_2))
    return merged.max(1)[0]


def read_script_from_file():
    script_path = osp.join(file_path, "./data_processing_script.txt")
    with open(script_path, "r") as f:
        script = f.readlines()
    return "".join(script)
