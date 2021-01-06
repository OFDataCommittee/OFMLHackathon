import os

from silc import Client

CLUSTER = True


def test_address():
    # get env var to set through client init
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]

    # client init should fail if SSDB not set
    c = Client(address=ssdb, cluster=CLUSTER)

    # check if SSDB was set anyway
    assert os.environ["SSDB"] == ssdb
