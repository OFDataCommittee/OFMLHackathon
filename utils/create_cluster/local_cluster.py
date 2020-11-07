from subprocess import Popen, TimeoutExpired, PIPE, SubprocessError, run
from time import sleep

def stop_cluster(n_nodes, port):
    """Stop a redis cluster and clear the files
    associated with it
    """
    import os
    redis = os.getenv('REDIS_INSTALL_PATH') + '/redis-cli'


    pids = []
    for i in range(n_nodes):
        cmd = redis + ' -p ' + str(port+i) + ' shutdown'
        pid = Popen([cmd], shell=True)
        pids.append(pid)

        sleep(1)
        fname = str(port+i) + ".log"
        if os.path.exists(fname):
            os.remove(fname)

        fname = str(port+i) + ".conf"
        if os.path.exists(fname):
            os.remove(fname)

    fname = 'dump.rdb'
    if os.path.exists(fname):
            os.remove(fname)

    return pids

def create_cluster(n_nodes, port):
    """Creates a cluster starting with port at
    127.0.0.1"""

    import os

    # Start servers
    host = '127.0.0.1'
    redis = os.getenv('REDIS_INSTALL_PATH') + '/redis-server'
    redisai = os.getenv('REDISAI_CPU_INSTALL_PATH') + '/redisai.so '
    pids = []

    for i in range(n_nodes):
        l_port = port + i
        cmd = redis + ' --port ' + str(l_port) + " --cluster-enabled yes --cluster-config-file "  + str(l_port) + ".conf --loadmodule " + \
            redisai + " --protected-mode no --loglevel notice "
        log_file = "--logfile "  + str(l_port) + ".log"
        cmd += log_file + ' '
        print(cmd)
        pid = Popen(cmd, shell=True)
        pids.append(pid)
    sleep(2)
    # Create cluster
    redis_cli = os.getenv('REDIS_INSTALL_PATH') + '/redis-cli'
    cluster_str=' '
    for i in range(n_nodes):
        cluster_str += '127.0.0.1:' + str(port+i) + ' '
    cmd = " ".join((redis_cli, "--cluster create", cluster_str, "--cluster-replicas 0"))
    print(cmd)
    proc = run([cmd], input="yes", encoding="utf-8", shell=True)
    if proc.returncode != 0:
        raise SubprocessError("Cluster could not be created!")
    else:
        print("Cluster has been setup!")

    return pids

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6379)
    parser.add_argument('--nodes', type=int, default=3)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()

    if(args.stop):
        stop_cluster(args.nodes, args.port)
    else:
        create_cluster(args.nodes, args.port)
