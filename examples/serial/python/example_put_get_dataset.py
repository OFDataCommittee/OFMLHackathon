import numpy as np

from silc import Client, Dataset

# Create two arrays
data_1 = np.random.randint(-10, 10, size=(10,10))
data_2 = np.random.randint(-10, 10, size=(20, 8, 2))
# Create a dataset to put
dataset = Dataset("test-dataset")
dataset.add_tensor("tensor_1", data_1)
dataset.add_tensor("tensor_2", data_2)

db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=False)

client.put_dataset(dataset)

rdataset = client.get_dataset("test-dataset")
rdata_1 = rdataset.get_tensor