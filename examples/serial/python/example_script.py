import numpy as np
import torch

from silc import Client

def two_to_one(data, data_2):
    """Sample torchscript script that returns the
    highest elements in both arguments

    Two inputs to one output
    """
    # return the highest element
    merged = torch.cat((data, data_2))
    return merged.max(1)[0]

data = np.array([[1, 2, 3, 4]])
data_2 = np.array([[5, 6, 7, 8]])

db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=False)

client.put_tensor("script-data-1", data)
client.put_tensor("script-data-2", data_2)
client.set_function("two-to-one", two_to_one)
client.run_script(
    "two-to-one",
    "two_to_one",
    ["script-data-1", "script-data-2"],
    ["script-multi-out-output"],
)
out = client.get_tensor("script-multi-out-output")