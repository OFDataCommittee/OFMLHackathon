# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
from smartredis.error import *


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
    with pytest.raises(RedisKeyError):
        c.get_dataset("not-a-key")


def test_bad_script_file(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(FileNotFoundError):
        c.set_script_from_file("key", "not-a-file")


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


def test_bad_device(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_script("key", "some_script", device="not-a-gpu")

#####
# Test type errors from bad parameter types to Client API calls

def test_bad_type_put_tensor(use_cluster):
    c = Client(None, use_cluster)
    array = np.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        c.put_tensor(42, array)
    with pytest.raises(TypeError):
        c.put_tensor("key", [1, 2, 3, 4])


def test_unsupported_type_put_tensor(use_cluster):
    """test an unsupported numpy type"""
    c = Client(None, use_cluster)
    data = np.array([1, 2, 3, 4]).astype(np.uint64)
    with pytest.raises(TypeError):
        c.put_tensor("key", data)


def test_bad_type_get_tensor(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.get_tensor(42)


def test_bad_type_delete_tensor(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.delete_tensor(42)


def test_bad_type_copy_tensor(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.copy_tensor(42, "newname")
    with pytest.raises(TypeError):
        c.copy_tensor("oldname", 42)


def test_bad_type_rename_tensor(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.rename_tensor(42, "newname")
    with pytest.raises(TypeError):
        c.rename_tensor("oldname", 42)


def test_bad_type_put_dataset(use_cluster):
    c = Client(None, use_cluster)
    array = np.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        c.put_dataset(array)


def test_bad_type_get_dataset(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.get_dataset(42)


def test_bad_type_delete_dataset(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.delete_dataset(42)


def test_bad_type_copy_dataset(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.copy_dataset(42, "dest")
    with pytest.raises(TypeError):
        c.copy_dataset("src", 42)


def test_bad_type_rename_dataset(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.rename_dataset(42, "oldkey")
    with pytest.raises(TypeError):
        c.rename_dataset("newkey", 42)


def test_bad_type_set_function(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_function(42, bad_function)
    with pytest.raises(TypeError):
        c.set_function("key", "not-a-function")
    with pytest.raises(TypeError):
        c.set_function("key", bad_function, 42)

def test_bad_type_set_function_multigpu(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_function_multigpu(42, bad_function, 0, 1)
    with pytest.raises(TypeError):
        c.set_function_multigpu("key", "not-a-function", 0, 1)
    with pytest.raises(TypeError):
        c.set_function_multigpu("key", bad_function, "not an integer", 1)
    with pytest.raises(TypeError):
        c.set_function_multigpu("key", bad_function, 0, "not an integer")
    with pytest.raises(ValueError):
        c.set_function_multigpu("key", bad_function, -1, 1) # invalid first GPU
    with pytest.raises(ValueError):
        c.set_function_multigpu("key", bad_function, 0, 0) # invalid num GPUs

def test_bad_type_set_script(use_cluster):
    c = Client(None, use_cluster)
    key = "key_for_script"
    script = "bad script but correct parameter type"
    device = "CPU"
    with pytest.raises(TypeError):
        c.set_script(42, script, device)
    with pytest.raises(TypeError):
        c.set_script(key, 42, device)
    with pytest.raises(TypeError):
        c.set_script(key, script, 42)

def test_bad_type_set_script_multigpu(use_cluster):
    c = Client(None, use_cluster)
    key = "key_for_script"
    script = "bad script but correct parameter type"
    first_gpu = 0
    num_gpus = 1
    with pytest.raises(TypeError):
        c.set_script_multigpu(42, script, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.set_script_multigpu(key, 42, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.set_script_multigpu(key, script, "not an integer", num_gpus)
    with pytest.raises(TypeError):
        c.set_script_multigpu(key, script, first_gpu, "not an integer")
    with pytest.raises(ValueError):
        c.set_script_multigpu(key, script, -1, num_gpus)
    with pytest.raises(ValueError):
        c.set_script_multigpu(key, script, first_gpu, 0)

def test_bad_type_set_script_from_file(use_cluster):
    c = Client(None, use_cluster)
    key = "key_for_script"
    scriptfile = "bad filename but correct parameter type"
    device = "CPU"
    with pytest.raises(TypeError):
        c.set_script_from_file(42, scriptfile, device)
    with pytest.raises(TypeError):
        c.set_script_from_file(key, 42, device)
    with pytest.raises(TypeError):
        c.set_script_from_file(key, scriptfile, 42)

def test_bad_type_set_script_from_file_multigpu(use_cluster):
    c = Client(None, use_cluster)
    key = "key_for_script"
    scriptfile = "bad filename but correct parameter type"
    first_gpu = 0
    num_gpus = 1
    with pytest.raises(TypeError):
        c.set_script_from_file_multigpu(42, scriptfile, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.set_script_from_file_multigpu(key, 42, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.set_script_from_file_multigpu(key, scriptfile, "not an integer", num_gpus)
    with pytest.raises(TypeError):
        c.set_script_from_file_multigpu(key, scriptfile, first_gpu, "not an integer")

def test_bad_type_get_script(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.get_script(42)


def test_bad_type_run_script(use_cluster):
    c = Client(None, use_cluster)
    key = "my_script"
    fn_name = "phred"
    inputs = ["list", "of", "strings"]
    outputs = ["another", "string", "list"]
    with pytest.raises(TypeError):
        c.run_script(42, fn_name, inputs, outputs)
    with pytest.raises(TypeError):
        c.run_script(key, 42, inputs, outputs)
    with pytest.raises(TypeError):
        c.run_script(key, fn_name, 42, outputs)
    with pytest.raises(TypeError):
        c.run_script(key, fn_name, inputs, 42)


def test_bad_type_run_script_multigpu(use_cluster):
    c = Client(None, use_cluster)
    key = "my_script"
    fn_name = "phred"
    inputs = ["list", "of", "strings"]
    outputs = ["another", "string", "list"]
    offset = 0
    first_gpu = 0
    num_gpus = 1
    with pytest.raises(TypeError):
        c.run_script_multigpu(42, fn_name, inputs, outputs, offset, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.run_script_multigpu(key, 42, inputs, outputs, offset, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.run_script_multigpu(key, fn_name, 42, outputs, offset, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.run_script_multigpu(key, fn_name, inputs, 42, offset, first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.run_script_multigpu(key, fn_name, inputs, outputs, "not an integer", first_gpu, num_gpus)
    with pytest.raises(TypeError):
        c.run_script_multigpu(key, fn_name, inputs, outputs, offset, "not an integer", num_gpus)
    with pytest.raises(TypeError):
        c.run_script_multigpu(key, fn_name, inputs, outputs, offset, first_gpu, "not an integer")
    with pytest.raises(ValueError):
        c.run_script_multigpu(key, fn_name, inputs, outputs, offset, -1, num_gpus)
    with pytest.raises(ValueError):
        c.run_script_multigpu(key, fn_name, inputs, outputs, offset, first_gpu, 0)


def test_bad_type_get_model(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.get_model(42)


def test_bad_type_set_model(mock_model, use_cluster):
    model = mock_model.create_torch_cnn()
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_model(42, model, "TORCH", "CPU")
    with pytest.raises(TypeError):
        c.set_model("simple_cnn", model, 42, "CPU")
    with pytest.raises(TypeError):
        c.set_model("simple_cnn", model, "UNSUPPORTED_ENGINE", "CPU")
    with pytest.raises(TypeError):
        c.set_model("simple_cnn", model, "TORCH", 42)
    with pytest.raises(TypeError):
        c.set_model("simple_cnn", model, "TORCH", "BAD_DEVICE")
    with pytest.raises(TypeError):
        c.set_model("simple_cnn", model, "TORCH", "CPU", batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model("simple_cnn", model, "TORCH", "CPU", min_batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model("simple_cnn", model, "TORCH", "CPU", tag=42)

def test_bad_type_set_model_multigpu(mock_model, use_cluster):
    c = Client(None, use_cluster)
    model = mock_model.create_torch_cnn()
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_model_multigpu(42, model, "TORCH", 0, 1)
    with pytest.raises(TypeError):
        c.set_model_multigpu("simple_cnn", model, 42, 0, 1)
    with pytest.raises(TypeError):
        c.set_model_multigpu("simple_cnn", model, "UNSUPPORTED_ENGINE", 0, 1)
    with pytest.raises(TypeError):
        c.set_model_multigpu("simple_cnn", model, "TORCH", "not an integer", 1)
    with pytest.raises(TypeError):
        c.set_model_multigpu("simple_cnn", model, "TORCH", 0, "not an integer")
    with pytest.raises(ValueError):
        c.set_model_multigpu("simple_cnn", model, "TORCH", -1, 1)
    with pytest.raises(ValueError):
        c.set_model_multigpu("simple_cnn", model, "TORCH", 0, 0)
    with pytest.raises(TypeError):
        c.set_model_multigpu("simple_cnn", model, "TORCH", 0, 1, batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model_multigpu("simple_cnn", model, "TORCH", 0, 1, min_batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model_multigpu("simple_cnn", model, "TORCH", 0, 1, tag=42)


def test_bad_type_set_model_from_file(use_cluster):
    modelfile = "bad filename but right parameter type"
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_model_from_file(42, modelfile, "TORCH", "CPU")
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", 42, "TORCH", "CPU")
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", modelfile, 42, "CPU")
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", modelfile, "UNSUPPORTED_ENGINE", "CPU")
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", modelfile, "TORCH", 42)
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", modelfile, "TORCH", "BAD_DEVICE")
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", modelfile, "TORCH", "CPU", batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", modelfile, "TORCH", "CPU", min_batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model_from_file("simple_cnn", modelfile, "TORCH", "CPU", tag=42)

def test_bad_type_set_model_from_file_multigpu(use_cluster):
    modelfile = "bad filename but right parameter type"
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu(42, modelfile, "TORCH", 0, 1)
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", 42, "TORCH", 0, 1)
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", modelfile, 42, 0, 1)
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", modelfile, "UNSUPPORTED_ENGINE", 0, 1)
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", modelfile, "TORCH", "not an integer", 1)
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", modelfile, "TORCH", 0, "not an integer")
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", modelfile, "TORCH", 0, 1, batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", modelfile, "TORCH", 0, 1, min_batch_size="not_an_integer")
    with pytest.raises(TypeError):
        c.set_model_from_file_multigpu("simple_cnn", modelfile, "TORCH", 0, 1, tag=42)

def test_bad_type_run_model(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.run_model(42)


def test_bad_type_run_model_multigpu(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.run_model_multigpu(42, 0, 0, 1)
    with pytest.raises(TypeError):
        c.run_model_multigpu("simple_cnn", "not an integer", 0, 1)
    with pytest.raises(TypeError):
        c.run_model_multigpu("simple_cnn", 0, "not an integer", 1)
    with pytest.raises(TypeError):
        c.run_model_multigpu("simple_cnn", 0, 0, "not an integer")
    with pytest.raises(ValueError):
        c.run_model_multigpu("simple_cnn", 0, -1, 1)
    with pytest.raises(ValueError):
        c.run_model_multigpu("simple_cnn", 0, 0, 0)

def test_bad_type_delete_model_multigpu(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.delete_model_multigpu(42, 0, 1)
    with pytest.raises(TypeError):
        c.delete_model_multigpu("simple_cnn",  "not an integer", 1)
    with pytest.raises(TypeError):
        c.delete_model_multigpu("simple_cnn", 0, "not an integer")
    with pytest.raises(ValueError):
        c.delete_model_multigpu("simple_cnn", -1, 1)
    with pytest.raises(ValueError):
        c.delete_model_multigpu("simple_cnn", 0, 0)

def test_bad_type_delete_script_multigpu(use_cluster):
    c = Client(None, use_cluster)
    script_name = "my_script"
    with pytest.raises(TypeError):
        c.delete_script_multigpu(42, 0, 1)
    with pytest.raises(TypeError):
        c.delete_script_multigpu(script_name,  "not an integer", 1)
    with pytest.raises(TypeError):
        c.delete_script_multigpu(script_name, 0, "not an integer")
    with pytest.raises(ValueError):
        c.delete_script_multigpu(script_name, -1, 1)
    with pytest.raises(ValueError):
        c.delete_script_multigpu(script_name, 0, 0)

def test_bad_type_tensor_exists(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.tensor_exists(42)


def test_bad_type_dataset_exists(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.dataset_exists(42)


def test_bad_type_model_exists(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.model_exists(42)


def test_bad_type_key_exists(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.key_exists(42)


def test_bad_type_poll_key(use_cluster):
    c = Client(None, use_cluster)
    name = "some_key"
    freq = 42
    num_tries = 42
    bogus = "bogus"
    with pytest.raises(TypeError):
        c.poll_key(42, freq, num_tries)
    with pytest.raises(TypeError):
        c.poll_key(name, bogus, num_tries)
    with pytest.raises(TypeError):
        c.poll_key(name, freq, bogus)


def test_bad_type_poll_tensor(use_cluster):
    c = Client(None, use_cluster)
    name = "some_key"
    freq = 42
    num_tries = 42
    bogus = "bogus"
    with pytest.raises(TypeError):
        c.poll_tensor(42, freq, num_tries)
    with pytest.raises(TypeError):
        c.poll_tensor(name, bogus, num_tries)
    with pytest.raises(TypeError):
        c.poll_tensor(name, freq, bogus)


def test_bad_type_poll_dataset(use_cluster):
    c = Client(None, use_cluster)
    name = "some_key"
    freq = 42
    num_tries = 42
    bogus = "bogus"
    with pytest.raises(TypeError):
        c.poll_dataset(42, freq, num_tries)
    with pytest.raises(TypeError):
        c.poll_dataset(name, bogus, num_tries)
    with pytest.raises(TypeError):
        c.poll_dataset(name, freq, bogus)


def test_bad_type_poll_model(use_cluster):
    c = Client(None, use_cluster)
    name = "some_key"
    freq = 42
    num_tries = 42
    bogus = "bogus"
    with pytest.raises(TypeError):
        c.poll_model(42, freq, num_tries)
    with pytest.raises(TypeError):
        c.poll_model(name, bogus, num_tries)
    with pytest.raises(TypeError):
        c.poll_model(name, freq, bogus)


def test_bad_type_set_data_source(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.set_data_source(42)


def test_bad_type_use_model_ensemble_prefix(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.use_model_ensemble_prefix("not a boolean")


def test_bad_type_use_tensor_ensemble_prefix(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.use_tensor_ensemble_prefix("not a boolean")


def test_bad_type_get_db_node_info(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.get_db_node_info("not a list")


def test_bad_type_get_db_cluster_info(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.get_db_cluster_info("not a list")


def test_bad_type_get_ai_info(use_cluster):
    c = Client(None, use_cluster)
    address = ["list", "of", "str"]
    key = "ai.info.key"
    with pytest.raises(TypeError):
        c.get_ai_info("not a list", key)
    with pytest.raises(TypeError):
        c.get_ai_info(address, 42)
    with pytest.raises(TypeError):
        c.get_ai_info(address, key, "not a boolean")


def test_bad_type_flush_db(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.flush_db("not a list")


def test_bad_type_config_get(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.config_get("timeout", 42)
    with pytest.raises(TypeError):
        c.config_get(42, "address")


def test_bad_type_config_set(use_cluster):
    c = Client(None, use_cluster)
    param = "timeout"
    value = "never"
    address = "127.0.0.1:6379"
    with pytest.raises(TypeError):
        c.config_set(42, value, address)
    with pytest.raises(TypeError):
        c.config_set(param, 42, address)
    with pytest.raises(TypeError):
        c.config_set(param, value, 42)


def test_bad_type_save(use_cluster):
    c = Client(None, use_cluster)
    with pytest.raises(TypeError):
        c.save("not a list")


#####
# Test type errors from bad parameter types to Dataset API calls

def test_bad_type_dataset():
    with pytest.raises(TypeError):
        d = Dataset(42)

def test_bad_type_add_tensor():
    d = Dataset("test-dataset")
    with pytest.raises(TypeError):
        d.add_tensor("test-tensor", [1, 2, 3])


def test_from_pybind_wrong_type():
    """A call to Dataset.set_data is made with the wrong
    type (i.e. not Pydataset).
    """
    input_param = Dataset("wrong_input_param")
    with pytest.raises(TypeError):
        d = Dataset.from_pybind(input_param)


def test_set_data_wrong_type():
    """A call to Dataset.set_data is made with the wrong
    type (i.e. not Pydataset).
    """
    d = Dataset("test_dataset")
    input_param = Dataset("wrong_input_param")
    with pytest.raises(TypeError):
        d.set_data(input_param)


def test_add_tensor_wrong_type():
    """A call to Dataset.add_tensor is made with the wrong type
    """
    d = Dataset("test_dataset")
    data = np.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        d.add_tensor(42, data)
    with pytest.raises(TypeError):
        d.add_tensor("tensorname", 42)

def test_get_tensor_wrong_type():
    """A call to Dataset.get_tensor is made with the wrong type
    """
    d = Dataset("test_dataset")
    with pytest.raises(TypeError):
        d.get_tensor(42)


def test_add_meta_scalar_wrong_type():
    """A call to Dataset.add_meta_scalar is made with the wrong type
    """
    d = Dataset("test_dataset")
    data = np.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        d.add_meta_scalar(42, 42)
    with pytest.raises(TypeError):
        d.add_meta_scalar("scalarname", data) # array, not scalar

def test_add_meta_string_wrong_type():
    """A call to Dataset.add_meta_string is made with the wrong type
    """
    d = Dataset("test_dataset")
    with pytest.raises(TypeError):
        d.add_meta_string(42, "metastring")
    with pytest.raises(TypeError):
        d.add_meta_string("scalarname", 42)


def test_get_meta_scalars_wrong_type():
    """A call to Dataset.get_meta_scalars is made with the wrong type
    """
    d = Dataset("test_dataset")
    with pytest.raises(TypeError):
        d.get_meta_scalars(42)


def test_get_meta_strings_wrong_type():
    """A call to Dataset.get_meta_strings is made with the wrong type
    """
    d = Dataset("test_dataset")
    with pytest.raises(TypeError):
        d.get_meta_strings(42)


####
# Utility functions

def bad_function(data):
    """Bad function which only raises an exception"""
    return False
