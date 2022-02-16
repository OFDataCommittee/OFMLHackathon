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

import numpy as np
import torch

from smartredis import Client

def two_to_one(data, data_2):
    """Sample torchscript script that returns the
    highest elements in both arguments

    Two inputs to one output
    """
    # return the highest element
    merged = torch.cat((data, data_2))
    return merged.max(1)[0]

# Connect a SmartRedis client to the Redis database
db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=True)

# Generate some test data to feed to the two_to_one function
data = np.array([[1, 2, 3, 4]])
data_2 = np.array([[5, 6, 7, 8]])

# Put the test data into the Redis database
client.put_tensor("script-data-1", data)
client.put_tensor("script-data-2", data_2)

# Put the function into the Redis database
client.set_function("two-to-one", two_to_one)

# Run the script using the test data
client.run_script(
    "two-to-one",
    "two_to_one",
    ["script-data-1", "script-data-2"],
    ["script-multi-out-output"],
)

# Retrieve the output of the test function
out = client.get_tensor("script-multi-out-output")