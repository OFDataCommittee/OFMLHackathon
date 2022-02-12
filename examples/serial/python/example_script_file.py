import os.path as osp
from smartredis import Client

# Construct a string holding the script file location
file_path = osp.dirname(osp.abspath(__file__))
script_path = osp.join(file_path, "./data_processing_script.txt")

# Connect to the Redis database
db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=True)

# Place the script in the database
client.set_script_from_file(
    "test-script-file", osp.join(file_path, "./data_processing_script.txt")
)
