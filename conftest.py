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
    return os.getenv('SILC_TEST_CLUSTER', "").lower() == 'true'

@pytest.fixture
def mock_data():
    return MockTestData

@pytest.fixture
def mock_model():
    return MockTestModel

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
