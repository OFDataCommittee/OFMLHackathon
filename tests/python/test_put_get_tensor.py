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
from smartredis import Client

# ----- Tests -----------------------------------------------------------


def test_1D_put_get(mock_data, use_cluster):
    """Test put/get_tensor for 1D numpy arrays"""

    client = Client(None, use_cluster)

    data = mock_data.create_data(10)
    send_get_arrays(client, data)


def test_2D_put_get(mock_data, use_cluster):
    """Test put/get_tensor for 2D numpy arrays"""

    client = Client(None, use_cluster)

    data = mock_data.create_data((10, 10))
    send_get_arrays(client, data)


def test_3D_put_get(mock_data, use_cluster):
    """Test put/get_tensor for 3D numpy arrays"""

    client = Client(None, use_cluster)

    data = mock_data.create_data((10, 10, 10))
    send_get_arrays(client, data)


# ------- Helper Functions -----------------------------------------------


def send_get_arrays(client, data):
    """Helper for put_get tests"""

    # put to database
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        client.put_tensor(key, array)
        assert client.tensor_exists(key)
        # get prefix, if it exists. Assumes the client is using
        # tensor prefix which is the default.
        sskeyin = os.environ.get("SSKEYIN", None)
        prefix = ""
        if sskeyin:
            prefix = sskeyin.split(",")[0] + "."
        assert client.key_exists(prefix + key)

    # get from database
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        rarray = client.get_tensor(key)
        np.testing.assert_array_equal(
            rarray, array, "Returned array from get_tensor not equal to sent tensor"
        )
