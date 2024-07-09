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

import os

import torch
import pytest
from os import environ
from smartredis import Client
from smartredis.error import *

test_gpu = environ.get("SMARTREDIS_TEST_DEVICE","cpu").lower() == "gpu"

def test_set_model(mock_model, context):
    model = mock_model.create_torch_cnn()
    c = Client(None, logger_name=context)
    c.set_model("simple_cnn", model, "TORCH", "CPU")
    returned_model = c.get_model("simple_cnn")
    assert model == returned_model


def test_set_model_from_file(mock_model, context):
    try:
        mock_model.create_torch_cnn(filepath="./torch_cnn.pt")
        c = Client(None, logger_name=context)
        c.set_model_chunk_size(1024 * 1024)
        c.set_model_from_file("file_cnn", "./torch_cnn.pt", "TORCH", "CPU")
        assert c.model_exists("file_cnn")
        returned_model = c.get_model("file_cnn")
        with open("./torch_cnn.pt", "rb") as f:
            model = f.read()
        assert model == returned_model
        c.delete_model("file_cnn")
        assert not c.model_exists("file_cnn")
    finally:
        os.remove("torch_cnn.pt")


def test_torch_inference(mock_model, context):
    # get model and set into database
    model = mock_model.create_torch_cnn()
    c = Client(None, logger_name=context)
    c.set_model("torch_cnn", model, "TORCH")

    # setup input tensor
    data = torch.rand(1, 1, 3, 3).numpy()
    c.put_tensor("torch_cnn_input", data)

    # run model and get output
    c.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
    out_data = c.get_tensor("torch_cnn_output")
    assert out_data.shape == (1, 1, 1, 1)

def test_batch_exceptions(mock_model, context):
    # get model and set into database
    mock_model.create_torch_cnn(filepath="./torch_cnn.pt")
    model = mock_model.create_torch_cnn()
    c = Client(None, logger_name=context)
    batch_size = 1
    min_batch_size = 1
    min_batch_timeout = 1
    with pytest.raises(RedisRuntimeError):
        c.set_model_from_file(
            "file_cnn", "./torch_cnn.pt", "TORCH", "CPU",
            batch_size=0, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_from_file(
            "file_cnn", "./torch_cnn.pt", "TORCH", "CPU",
            batch_size=0, min_batch_size=min_batch_size, min_batch_timeout=0
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_from_file(
            "file_cnn", "./torch_cnn.pt", "TORCH", "CPU",
            batch_size=batch_size, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_from_file_multigpu(
            "file_cnn", "./torch_cnn.pt", "TORCH", 1, 1,
            batch_size=0, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_from_file_multigpu(
            "file_cnn", "./torch_cnn.pt", "TORCH", 1, 1,
            batch_size=0, min_batch_size=min_batch_size, min_batch_timeout=0
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_from_file_multigpu(
            "file_cnn", "./torch_cnn.pt", "TORCH", 1, 1,
            batch_size=batch_size, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model(
            "file_cnn", model, "TORCH", "CPU",
            batch_size=0, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model(
            "file_cnn", model, "TORCH", "CPU",
            batch_size=0, min_batch_size=min_batch_size, min_batch_timeout=0
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model(
            "file_cnn", model, "TORCH", "CPU",
            batch_size=batch_size, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_multigpu(
            "file_cnn", model, "TORCH", 1, 1,
            batch_size=0, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_multigpu(
            "file_cnn", model, "TORCH", 1, 1,
            batch_size=0, min_batch_size=min_batch_size, min_batch_timeout=0
        )
    with pytest.raises(RedisRuntimeError):
        c.set_model_multigpu(
            "file_cnn", model, "TORCH", 1, 1,
            batch_size=batch_size, min_batch_size=0, min_batch_timeout=min_batch_timeout
        )

def test_batch_warning_set_model_from_file(mock_model, context, capfd):
    # get model and set into database
    mock_model.create_torch_cnn(filepath="./torch_cnn.pt")
    c = Client(None, logger_name=context)
    c.set_model_from_file(
        "file_cnn", "./torch_cnn.pt", "TORCH", "CPU",
        batch_size=1, min_batch_size=1, min_batch_timeout=0
    )
    captured = capfd.readouterr()
    assert "WARNING" in captured.err

@pytest.mark.skipif(
    not test_gpu,
    reason="SMARTREDIS_TEST_DEVICE does not specify 'gpu'"
)
def test_batch_warning_set_model_from_file_multigpu(mock_model, context, capfd):
    # get model and set into database
    mock_model.create_torch_cnn(filepath="./torch_cnn.pt")
    c = Client(None, logger_name=context)
    c.set_model_from_file_multigpu(
        "file_cnn", "./torch_cnn.pt", "TORCH", 1, 1,
        batch_size=1, min_batch_size=1, min_batch_timeout=0
    )
    captured = capfd.readouterr()
    assert "WARNING" in captured.err

def test_batch_warning_set_model(mock_model, context, capfd):
    # get model and set into database
    model = mock_model.create_torch_cnn()
    c = Client(None, logger_name=context)
    c.set_model(
        "file_cnn", model, "TORCH", "CPU",
        batch_size=1, min_batch_size=1, min_batch_timeout=0
    )
    captured = capfd.readouterr()
    assert "WARNING" in captured.err

@pytest.mark.skipif(
    not test_gpu,
    reason="SMARTREDIS_TEST_DEVICE does not specify 'gpu'"
)
def test_batch_warning_set_model_multigpu(mock_model, context, capfd):
    # get model and set into database
    model = mock_model.create_torch_cnn()
    c = Client(None, logger_name=context)
    c.set_model_multigpu(
        "file_cnn", model, "TORCH", 1, 1,
        batch_size=1, min_batch_size=1, min_batch_timeout=0
    )
    captured = capfd.readouterr()
    assert "WARNING" in captured.err
