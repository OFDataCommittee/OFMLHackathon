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
from smartredis.error import RedisConnectionError, RedisReplyError


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
        c.run_script(
            "bad-function", "not-a-function-in-script", ["bad-func-tensor"], ["output"]
        )


def test_wrong_model_name(mock_data, mock_model, use_cluster):
    """User requests to run a model that is not there"""

    data = mock_data.create_data(1)

    model = mock_model.create_torch_cnn()
    c = Client(None, use_cluster)
    c.set_model("simple_cnn", model, "TORCH", "CPU")
    c.put_tensor("input", data[0])
    with pytest.raises(RedisReplyError):
        c.run_model("wrong_cnn", ["input"], ["output"])


def test_wrong_model_name_from_file(mock_data, mock_model, use_cluster):
    """User requests to run a model that is not there
    that was loaded from file."""

    try:
        data = mock_data.create_data(1)
        mock_model.create_torch_cnn(filepath="./torch_cnn.pt")
        c = Client(None, use_cluster)
        c.set_model_from_file("simple_cnn_from_file", "./torch_cnn.pt", "TORCH", "CPU")
        c.put_tensor("input", data[0])
        with pytest.raises(RedisReplyError):
            c.run_model("wrong_cnn", ["input"], ["output"])
    finally:
        os.remove("torch_cnn.pt")

def test_set_data_wrong_type():
    """A call to Dataset.set_data is made with the wrong
    type (i.e. not Pydataset).
    """
    d = Dataset("test_dataset")
    input_param = Dataset("wrong_input_param")
    with pytest.raises(TypeError):
        d.set_data(input_param)

def test_from_pybind_wrong_type():
    """A call to Dataset.set_data is made with the wrong
    type (i.e. not Pydataset).
    """
    input_param = Dataset("wrong_input_param")
    with pytest.raises(TypeError):
        d = Dataset.from_pybind(input_param)

def bad_function(data):
    """Bad function which only raises an exception"""
    return False
