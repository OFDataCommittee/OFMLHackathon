import os

from silc import Client


def test_address(use_cluster):
    # get env var to set through client init
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]

    # client init should fail if SSDB not set
    c = Client(address=ssdb, cluster=use_cluster)

    # check if SSDB was set anyway
    assert os.environ["SSDB"] == ssdb
