import os

def cluster():
    cluster_env_var = os.environ["SILC_TEST_CLUSTER"]
    cluster = False
    if cluster_env_var.lower() == "true":
        cluster = True
    return cluster