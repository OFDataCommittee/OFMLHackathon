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

import io

import torch
import torch.nn as nn

from smartredis import Client

# Taken from https://pytorch.org/docs/master/generated/torch.jit.trace.html
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

net = Net()
example_forward_input = torch.rand(1, 1, 3, 3)

# Trace a module (implicitly traces `forward`) and construct a
# `ScriptModule` with a single `forward` method
module = torch.jit.trace(net, example_forward_input)

# Create a buffer of the traced model
buffer = io.BytesIO()
torch.jit.save(module, buffer)
model = buffer.getvalue()

# Connect a SmartRedis client and set the model in the database
db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=True, logger_name="example_model_torch.py")
client.set_model("torch_cnn", model, "TORCH", "CPU")

# Retrieve the model and verify that the retrieved
# model matches the original model.
returned_model = client.get_model("torch_cnn")
assert model == returned_model

# Setup input tensor
data = torch.rand(1, 1, 3, 3).numpy()
client.put_tensor("torch_cnn_input", data)

# Run model and get output
client.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
out_data = client.get_tensor("torch_cnn_output")