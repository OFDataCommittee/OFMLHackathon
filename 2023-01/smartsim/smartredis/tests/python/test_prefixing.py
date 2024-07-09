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

import numpy as np
import os

from smartredis import Client, Dataset

def test_prefixing(context, monkeypatch):
    # configure prefix variables
    monkeypatch.setenv("SSKEYOUT", "prefix_test")
    monkeypatch.setenv("SSKEYIN", "prefix_test,prefix_ignore")

    # Set up client
    c = Client(logger_name=context)
    c.use_dataset_ensemble_prefix(True)
    c.use_tensor_ensemble_prefix(True)
    c.set_data_source("prefix_test")

    # Create Dataset
    d = Dataset("test_dataset")
    data = np.uint16([1, 2, 3, 4])
    d.add_tensor("dataset_tensor", data)
    c.put_dataset(d)
    c.put_tensor("test_tensor", data)

    # Validate keys to see whether prefixing was applied properly
    assert c.dataset_exists("test_dataset")
    assert c.key_exists("prefix_test.{test_dataset}.meta")
    assert not c.key_exists("test_dataset")
    assert c.tensor_exists("test_tensor")
    assert c.key_exists("prefix_test.test_tensor")
    assert not c.key_exists("test_tensor")

def test_model_prefixing(mock_model, context, monkeypatch):
    # configure prefix variables
    monkeypatch.setenv("SSKEYOUT", "prefix_test")
    monkeypatch.setenv("SSKEYIN", "prefix_test,prefix_ignore")

    # Set up client
    c = Client(logger_name=context)
    c.use_model_ensemble_prefix(True)
    c.set_data_source("prefix_test")

    # Create model
    model = mock_model.create_torch_cnn()
    c.set_model("simple_cnn", model, "TORCH", "CPU")

    # Validate keys to see whether prefixing was applied properly
    assert c.model_exists("simple_cnn")
    assert not c.key_exists("simple_cnn")


def test_list_prefixing(context, monkeypatch):
    # configure prefix variables
    monkeypatch.setenv("SSKEYOUT", "prefix_test")
    monkeypatch.setenv("SSKEYIN", "prefix_test,prefix_ignore")

    # Set up client
    c = Client(logger_name=context)
    c.use_list_ensemble_prefix(True)
    c.set_data_source("prefix_test")

    # Build datasets
    num_datasets = 4
    original_datasets = [create_dataset(f"dataset_{i}") for i in range(num_datasets)]

    # Make sure the list is cleared
    list_name = "dataset_test_list"
    c.delete_list(list_name)

    # Put datasets into the list
    for i in range(num_datasets):
        c.put_dataset(original_datasets[i])
        c.append_to_list(list_name, original_datasets[i])

    # Validate keys to see whether prefixing was applied properly
    assert c.key_exists("prefix_test.dataset_test_list")
    assert not c.key_exists("dataset_test_list")

# ------------ helper functions ---------------------------------

def create_dataset(name):
    array = np.array([1, 2, 3, 4])
    string = "test_meta_strings"
    scalar = 7

    dataset = Dataset(name)
    dataset.add_tensor("test_array", array)
    dataset.add_meta_string("test_string", string)
    dataset.add_meta_scalar("test_scalar", scalar)
    return dataset