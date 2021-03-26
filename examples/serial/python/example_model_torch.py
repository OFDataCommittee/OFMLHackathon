import io

import torch
import torch.nn as nn

from silc import Client

# taken from https://pytorch.org/docs/master/generated/torch.jit.trace.html
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

buffer = io.BytesIO()
torch.jit.save(module, buffer)
model = buffer.getvalue()

db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=False)
client.set_model("torch_cnn", model, "TORCH", "CPU")
returned_model = client.get_model("torch_cnn")
assert model == returned_model

# setup input tensor
data = torch.rand(1, 1, 3, 3).numpy()
client.put_tensor("torch_cnn_input", data)

# run model and get output
client.run_model("torch_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
out_data = client.get_tensor("torch_cnn_output")