import inspect
import os.path as osp

import numpy as np
import torch

from silc import Client

CLUSTER = True
file_path = osp.dirname(osp.abspath(__file__))


def example_run_script():
    data = np.array([[1, 2, 3, 4, 5]])

    c = Client(None, CLUSTER, False)
    c.put_tensor("script-test-data", data)
    c.set_function("one-to-one", one_to_one)
    c.run_script("one-to-one", "one_to_one", ["script-test-data"], ["script-test-out"])
    out = c.get_tensor("script-test-out")
    assert out == 5


def example_run_script_multi():
    data = np.array([[1, 2, 3, 4]])
    data_2 = np.array([[5, 6, 7, 8]])

    c = Client(None, CLUSTER, False)
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

if __name__ == "__main__":
    example_run_script()
    example_run_script_multi()
    print("SILC run script method example complete.")
