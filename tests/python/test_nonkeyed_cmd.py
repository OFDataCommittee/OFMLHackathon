import os
import numpy as np
import pytest

from smartredis import Client
from smartredis.error import RedisReplyError


def test_dbnode_info_command(use_cluster):
    # get env var to set through client init
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


def test_flushdb_command(use_cluster):
    # get env var to set through client init
    ssdb = os.environ["SSDB"]
    if use_cluster:
        addresses = ssdb.split(",")
    else:
        addresses = [ssdb]
    del os.environ["SSDB"]

    # client init should fail if SSDB not set
    client = Client(address=ssdb, cluster=use_cluster)

    # add key to client via put_tensor
    tensor = np.array([1, 2])
    client.put_tensor("test_copy", tensor)

    reply = client.flush_db(addresses)
    with pytest.raises(RedisReplyError):
        client.get_tensor("test_copy")
    assert reply == "OK"


def test_config_set_get_command(use_cluster):
    # get env var to set through client init
    ssdb = os.environ["SSDB"]

    del os.environ["SSDB"]
    # client init should fail if SSDB not set
    client = Client(address=ssdb, cluster=use_cluster)

    value = "6000"
    set_reply = client.config_set("lua-time-limit", value, ssdb)
    assert set_reply == "OK"

    get_reply = client.config_get("lua-time-limit", ssdb)
    assert len(get_reply) > 0
    assert get_reply["lua-time-limit"] == value