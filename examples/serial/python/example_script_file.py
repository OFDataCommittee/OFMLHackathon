import os.path as osp

from silc import Client

file_path = osp.dirname(osp.abspath(__file__))

script_path = osp.join(file_path, "./data_processing_script.txt")
with open(script_path, "r") as f:
    script = f.readlines()
sent_script = "".join(script)

db_address = "127.0.0.1:6379"
client = Client(address=db_address, cluster=False)
client.set_script_from_file(
    "test-script-file", osp.join(file_path, "./data_processing_script.txt")
)
