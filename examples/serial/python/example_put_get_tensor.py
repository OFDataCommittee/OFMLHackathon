import numpy as np
from smartredis import Client

# Connect a SmartRedis client to Redis database
db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=False)

# Send a 2D tensor to the database
key = "2D_array"
array = np.random.randint(-10, 10, size=(10, 10))
client.put_tensor(key, array)

# Retrieve the tensor
returned_array = client.get_tensor("2D_array")