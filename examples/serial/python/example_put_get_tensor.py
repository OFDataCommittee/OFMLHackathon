from silc import Client
import numpy as np

# initialize the client (keep connections alive)
db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=False)

# Send a 2D Tensor
key = "2D_array"
array = np.random.randint(-10, 10, size=(10, 10))
client.put_tensor(key, array)

# Get the 2D Tensor
returned_array = client.get_tensor("2D_array")