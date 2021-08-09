from subprocess import Popen, TimeoutExpired, PIPE, SubprocessError, run
from os import environ
import os

def get_ip_from_host(host):
    ping_out = ping_host(host)
    found = False
    # break loop when the hostname is found in
    # the ping output
    for item in ping_out.split():
        if found:
            return item.split("(")[1].split(")")[0]
        if item == host:
            found = True

def launch_db(host, port, dpn):
    """
    set cluster log file
    set log file
    """

    redis = environ['REDIS_INSTALL_PATH'] + '/redis-server '
    redisai = "--loadmodule " +  environ['REDISAI_CPU_INSTALL_PATH'] + "/redisai.so "
    if(dpn < 3):
        cmd = "srun -N 1 -n 1 " + redis + "./smartredisdb.conf " + redisai + "--cluster-enabled yes  " + "--bind " + get_ip_from_host(host) + " "
        log_file = "--logfile " + host + "-" + str(port) + ".log"
        cluster_file = "--cluster-config-file nodes-" + host + "-" + str(port) + ".conf"
        cmd += log_file + " "
        cmd += cluster_file + " "
        print(cmd)
        pid = Popen(cmd, shell=True)
        return pid
    else:
        conf_file = open("run_orc.conf","w")
        for i in range(dpn):
            dir = os.getcwd()
            cmd = redis + dir+"/smartredisdb.conf " + "--port " + str(port+i) + " " + redisai + "--cluster-enabled yes " +  "--bind " + get_ip_from_host(host) + " "
            log_file = "--logfile " + host + "-" + str(port+i) + ".log"
            cluster_file = "--cluster-config-file nodes-" + host + "-" + str(port+i) + ".conf"
            cmd += log_file + " "
            cmd += cluster_file + " "
            conf_str = str(i) + " " + cmd + "\n"
            conf_file.write(conf_str)
        conf_file.close()
        dpn_cmd = "srun -N 1 -n " + str(dpn) + " --multi-prog run_orc.conf"
        print(dpn_cmd)
        pid = Popen(dpn_cmd , shell=True)
        time.sleep(1)
        return pid

def create_cluster(nodes, port, dpn):
    # TODO change to support list instead of node string when put into smartsim
    cluster_str = ""
    ssdb_str = ""
    for node in nodes.split(","):
        for i in range(dpn):
            node_ip = get_ip_from_host(node)
            node_ip += ":" + str(port+i)
            cluster_str += node_ip + " "
            ssdb_str += node_ip +  ","
    ssdb_str = ssdb_str[0:-1]

    # call cluster command
    keydb_cli = environ['REDIS_INSTALL_PATH'] + '/redis-cli '
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


def create_node_string(nodes, node_prefix, node_pad):
    """'30-40, 80-90'"""
    node_string = ""
    for nodepair in nodes.split(","):
        split = nodepair.split("-")
        start = int(split[0])
        end = int(split[1]) + 1
        for node_num in range(start, end):
            node_string += node_prefix + f'{node_num:0{node_pad}}' + ','
            #if node_num > 99:
            #    node_string += "nid00" + str(node_num) + ","
            #elif node_num < 10:
            #    node_string += "nid0000" + str(node_num) + ","
            #else:
            #    node_string += "nid000" + str(node_num) + ","
    return node_string.strip(",")

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=str)
    parser.add_argument('--port', type=int, default=6379)
    parser.add_argument('--dpn', type=int, default=1)
    # node string parameters
    parser.add_argument('--node-prefix', type=str, default='nid')
    parser.add_argument('--node-pad', type=int, default=5)
    args = parser.parse_args()

    nodes = create_node_string(args.nodes, args.node_prefix, args.node_pad)
    for node in nodes.split(","):
        pid = launch_db(node, args.port, args.dpn)
    time.sleep(5)
    create_cluster(nodes, args.port, args.dpn)
