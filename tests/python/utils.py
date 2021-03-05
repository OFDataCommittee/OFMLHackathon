import os

def use_cluster():
    cluster_env_var = os.environ["SILC_TEST_CLUSTER"]
    if cluster_env_var.lower() == "true":
        return True
    return False
