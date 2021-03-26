import os

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

db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=False)

try:
    net = Net()
    example_forward_input = torch.rand(1, 1, 3, 3)
    # Trace a module (implicitly traces `forward`) and construct a
    # `ScriptModule` with a single `forward` method
    module = torch.jit.trace(net, example_forward_input)
    torch.jit.save(module, "./torch_cnn.pt")
    client.set_model_from_file("file_cnn", "./torch_cnn.pt", "TORCH", "CPU")
    data = torch.rand(1, 1, 3, 3).numpy()
    client.put_tensor("torch_cnn_input", data)

    # run model and get output
    client.run_model("file_cnn", inputs=["torch_cnn_input"], outputs=["torch_cnn_output"])
    out_data = client.get_tensor("torch_cnn_output")
finally:
    os.remove("torch_cnn.pt")
