from subprocess import Popen, TimeoutExpired, PIPE, SubprocessError, run
from os import environ

def launch_db(host, port):
    """
    set cluster log file
    set log file
    """
    redis = "keydb-server "
    redisai = "--loadmodule " +  environ['REDISAI_CPU_INSTALL_PATH'] + "/redisai.so "
    cmd = "srun -N 1 -n 1 " + redis + "./silcdb.conf " + redisai + "--cluster-enabled yes  "
    log_file = "--logfile " + host + "-" + str(port) + ".log"
    cluster_file = "--cluster-config-file nodes-" + host + "-" + str(port) + ".conf"
    cmd += log_file + " "
    cmd += cluster_file + " "
    print(cmd)
    pid = Popen(cmd, shell=True)
    return pid


def get_ip_from_host(host):
    ping_out = ping_host(host)
    found = False
    print(host)
    print(ping_out)
    print('here')
    # break loop when the hostname is found in
    # the ping output
    for item in ping_out.split():
        print(item)
        if found:
            print('Found!')
            return item.split("(")[1].split(")")[0]
        if item == host:
            found = True


def create_cluster(nodes, port):
    # TODO change to support list instead of node string when put into smartsim
    cluster_str = ""
    ssdb_str = ""
    for node in nodes.split(","):
        node_ip = get_ip_from_host(node)
        node_ip += ":" + str(port)
        cluster_str += node_ip + " "
        ssdb_str += node_ip +  ","
    ssdb_str = ssdb_str[0:-1]

    # call cluster command
    keydb_cli = "keydb-cli "
    cmd = " ".join((keydb_cli, "--cluster create", cluster_str, "--cluster-replicas 0"))
    print(cmd)
    proc = run([cmd], input="yes", encoding="utf-8", shell=True)
    if proc.returncode != 0:
        raise SubprocessError("Cluster could not be created!")
    else:
        print("Cluster has been setup!")
    print('Export SSDB= ')
    print(ssdb_str)



def ping_host(hostname):
    proc = Popen("ping -c 1 " + hostname, stderr=PIPE, stdout=PIPE, shell=True)
    try:
        output, errs = proc.communicate(timeout=15)
        return output.decode("utf-8")
    except TimeoutExpired:
        proc.kill()
        output, errs = proc.communicate()


def create_node_string(nodes):
    """'30-40, 80-90'"""
    node_string = ""
    for nodepair in nodes.split(","):
        split = nodepair.split("-")
        start = int(split[0])
        end = int(split[1]) + 1
        for node_num in range(start, end):
            if node_num > 99:
                node_string += "nid00" + str(node_num) + ","
            elif node_num < 10:
                node_string += "nid0000" + str(node_num) + ","
            else:
                node_string += "nid000" + str(node_num) + ","
    return node_string.strip(",")

def create_local_cluster(n_nodes, port):
    """Creates a cluster starting with port at
    127.0.0.1"""

    import os

    host = '127.0.0.1'
    redis = os.getenv('REDIS_INSTALL_PATH') + '/redis-server'
    redisai = os.getenv('REDISAI_CPU_INSTALL_PATH') + '/redisai.so '
    pids = []

    for i in range(n_nodes):
        l_port = port + i
        cmd = redis + ' --port ' + str(l_port) + ' --cluster-enabled yes --loadmodule ' + \
            redisai + "--cluster-enabled yes --protected-mode no --loglevel notice "
        log_file = "--logfile " + host + "-" + str(l_port) + ".log"
        cluster_file = "--cluster-config-file nodes-" + host + "-" + str(l_port) + ".conf"
        print(cmd)
        pid = Popen(cmd, shell=True)
        pids.append(pid)
    return pids

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=str)
    parser.add_argument('--port', type=int, default=6379)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--n_local_nodes', type=int, default=3)
    args = parser.parse_args()

    if args.local:
        create_local_cluster(args.n_local_nodes,
                             args.port)
    else:
        nodes = create_node_string(args.nodes)
        for node in nodes.split(","):
            pid = launch_db(node, args.port)
        time.sleep(5)
        create_cluster(nodes, args.port)
