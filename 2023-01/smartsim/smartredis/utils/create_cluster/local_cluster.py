# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    test_device = os.environ.get("SMARTREDIS_TEST_DEVICE","cpu").lower()
    redisai = os.getenv(f'REDISAI_{test_device.upper()}_INSTALL_PATH') + '/redisai.so '
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
