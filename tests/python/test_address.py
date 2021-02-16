import os

from silc import Client
import get_cluster_env

CLUSTER = get_cluster_env.cluster()

def test_address():
    # get env var to set through client init
    ssdb = os.environ["SSDB"]
    del os.environ["SSDB"]

    # client init should fail if SSDB not set
    c = Client(address=ssdb, cluster=CLUSTER)

    # check if SSDB was set anyway
    assert os.environ["SSDB"] == ssdb
