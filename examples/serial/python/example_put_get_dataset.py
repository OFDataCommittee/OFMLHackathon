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
client = Client(address=db_address, cluster=True)

# Place the DataSet into the database
client.put_dataset(dataset)

# Retrieve the DataSet from the database
rdataset = client.get_dataset("test-dataset")

# Retrieve a tensor from inside of the fetched
# DataSet
rdata_1 = rdataset.get_tensor("tensor_1")