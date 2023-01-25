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
import numpy as np
import torch
import torch.nn as nn
import io
import os
import random
import string

dtypes = [
    np.float64,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
]

metadata_scalar_dtypes = [
    np.float32,
    np.float64,
    np.int32,
    np.int64,
    np.uint32,
    np.uint64,
]

@pytest.fixture
def use_cluster():
    return os.getenv('SMARTREDIS_TEST_CLUSTER', "").lower() == 'true'

@pytest.fixture
def mock_data():
    return MockTestData

@pytest.fixture
def mock_model():
    return MockTestModel

@pytest.fixture
def context(request):
    return request.node.name

class MockTestData:

    @staticmethod
    def create_data(shape):
        """Helper for creating numpy data"""

        data = []
        for dtype in dtypes:
            array = np.random.randint(-10, 10, size=shape).astype(dtype)
            data.append(array)
        return data

    @staticmethod
    def create_metadata_scalars(length):
        """Helper for creating numpy data"""

        data = []
        for dtype in metadata_scalar_dtypes:
            array = np.random.randint(-10, 10, size=length).astype(dtype)
            data.append(array)
        return data

    @staticmethod
    def create_metadata_strings(length):
        """Helper for creating list of strings"""

        data = []
        for _ in range(length):
            meta_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))
            data.append(meta_string)
        return data


# taken from https://pytorch.org/docs/master/generated/torch.jit.trace.html
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

class MockTestModel:

    @staticmethod
    def create_torch_cnn(filepath=None):
        """Create a torch CNN for testing purposes

        Jit traces the torch Module for storage in RedisAI
        Function either saves to a file or returns a byte string
        """
        n = Net()
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

        if filepath:
            torch.jit.save(module, filepath)
            return
        else:
            # save model into an in-memory buffer then string
            buffer = io.BytesIO()
            torch.jit.save(module, buffer)
            str_model = buffer.getvalue()
            return str_model
