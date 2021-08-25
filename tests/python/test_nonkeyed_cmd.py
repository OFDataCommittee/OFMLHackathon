import os
import numpy as np

from smartredis import Client


def test_dbnode_info_command(use_cluster):
    # get env var to set through client init
    # if cluster, only test first DB
    ssdb = os.environ["SSDB"]
    if use_cluster:
        db_info_addr = ssdb.split(",")
    else:
        db_info_addr = [ssdb]
    del os.environ["SSDB"]

    # client init should fail if SSDB not set
    client = Client(address=ssdb, cluster=use_cluster)

    info = client.get_db_node_info(db_info_addr)

    assert len(info) > 0

    for db in info:
        for key in db:
            print("\n")
            print(key, db[key])


def test_flushall_command(use_cluster):
    # get env var to set through client init
    # if cluster, only test first DB
    ssdb = os.environ["SSDB"]

    del os.environ["SSDB"]
    # client init should fail if SSDB not set
    client = Client(address=ssdb, cluster=use_cluster)

    # add key to client via put_tensor
    tensor = np.array([1, 2])
    client.put_tensor("test_copy", tensor)

    reply = client.flush_all_db()
    assert not (client.key_exists("test_copy"))
    assert reply == "OK"
