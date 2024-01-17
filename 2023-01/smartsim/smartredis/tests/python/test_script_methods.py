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

import pytest
import inspect
import os.path as osp
from os import environ
import numpy as np
import torch
from smartredis import Client

file_path = osp.dirname(osp.abspath(__file__))


test_gpu = environ.get("SMARTREDIS_TEST_DEVICE","cpu").lower() == "gpu"

def test_set_get_function(context):
    c = Client(None, logger_name=context)
    c.set_function("test-set-function", one_to_one)
    script = c.get_script("test-set-function")
    sent_script = inspect.getsource(one_to_one)
    assert script == sent_script


def test_set_get_script(context):
    c = Client(None, logger_name=context)
    sent_script = read_script_from_file()
    c.set_script("test-set-script", sent_script)
    script = c.get_script("test-set-script")
    assert sent_script == script


def test_set_script_from_file(context):
    sent_script = read_script_from_file()
    c = Client(None, logger_name=context)
    c.set_script_from_file(
        "test-script-file", osp.join(file_path, "./data_processing_script.txt")
    )
    assert c.model_exists("test-script-file")
    returned_script = c.get_script("test-script-file")
    assert sent_script == returned_script
    c.delete_script("test-script-file")
    assert not c.model_exists("test-script-file")


def test_run_script_str(context):
    data = np.array([[1, 2, 3, 4, 5]])

    c = Client(None, logger_name=context)
    c.put_tensor("script-test-data", data)
    c.set_function("one-to-one", one_to_one)
    c.run_script("one-to-one", "one_to_one", "script-test-data", "script-test-out")
    out = c.get_tensor("script-test-out")
    assert out == 5


def test_run_script_list(context):
    data = np.array([[1, 2, 3, 4]])
    data_2 = np.array([[5, 6, 7, 8]])

    c = Client(None, logger_name=context)
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

@pytest.mark.skipif(
    not test_gpu,
    reason="SMARTREDIS_TEST_DEVICE does not specify 'gpu'"
)
def test_run_script_multigpu_str(use_cluster, context):
    data = np.array([[1, 2, 3, 4, 5]])

    c = Client(None, use_cluster, logger_name=context)
    c.put_tensor("script-test-data", data)
    c.set_function_multigpu("one-to-one", one_to_one, 0, 2)
    c.run_script_multigpu("one-to-one", "one_to_one", "script-test-data", "script-test-out", 0, 0, 2)
    out = c.get_tensor("script-test-out")
    assert out == 5

@pytest.mark.skipif(
    not test_gpu,
    reason="SMARTREDIS_TEST_DEVICE does not specify 'gpu'"
)
def test_run_script_multigpu_list(use_cluster, context):
    data = np.array([[1, 2, 3, 4]])
    data_2 = np.array([[5, 6, 7, 8]])

    c = Client(None, use_cluster, logger_name=context)
    c.put_tensor("srpt-multi-out-data-1", data)
    c.put_tensor("srpt-multi-out-data-2", data_2)
    c.set_function_multigpu("two-to-one", two_to_one, 0, 2)
    c.run_script_multigpu(
        "two-to-one",
        "two_to_one",
        ["srpt-multi-out-data-1", "srpt-multi-out-data-2"],
        ["srpt-multi-out-output"],
        0,
        0,
        2
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
