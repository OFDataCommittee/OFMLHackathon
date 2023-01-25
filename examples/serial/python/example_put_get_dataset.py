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

from smartredis import Client, Dataset

# Create two arrays to store in the DataSet
data_1 = np.random.randint(-10, 10, size=(10,10))
data_2 = np.random.randint(-10, 10, size=(20, 8, 2))

# Create a DataSet object and add the two sample tensors
dataset = Dataset("test-dataset")
dataset.add_tensor("tensor_1", data_1)
dataset.add_tensor("tensor_2", data_2)

# Connect SmartRedis client to Redis database
db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=True, logger_name="example_put_get_dataset.py")

# Place the DataSet into the database
client.put_dataset(dataset)

# Retrieve the DataSet from the database
rdataset = client.get_dataset("test-dataset")

# Retrieve a tensor from inside of the fetched
# DataSet
rdata_1 = rdataset.get_tensor("tensor_1")