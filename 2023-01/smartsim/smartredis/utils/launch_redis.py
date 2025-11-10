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

from subprocess import Popen, SubprocessError, run, DEVNULL
from time import sleep
import argparse
import os
import pathlib

def check_availability(n_nodes, port, udsport):
    """Repeat a command until it is successful
    """
    num_tries = 5
    is_uds = udsport is not None
    if is_uds:
        n_nodes = 1
    cicd = os.getenv('SR_CICD_EXECUTION')
    is_cicd = False if cicd is None else cicd.lower() == "true"
    if is_cicd:
        rediscli = 'redis-cli'
    else:
        rediscli = (
            pathlib.Path(__file__).parent.parent
            / "third-party/redis/src/redis-cli"
        ).resolve()
    for i in range(n_nodes):
        connection = f"-s {udsport}" if is_uds else f"-p {port + i}"
        set_cmd = f"{rediscli} {connection} set __test__ __test__"
        del_cmd = f"{rediscli} {connection} del __test__"
        command_succeeded = False
        for _ in range(num_tries):
            try:
                run(set_cmd.split(), shell=False, stdout=DEVNULL, stderr=DEVNULL)
                run(del_cmd.split(), shell=False, stdout=DEVNULL, stderr=DEVNULL)
                command_succeeded = True
                break
            except Exception:
                # That try failed, so just retry
                sleep(5)
        if not command_succeeded:
            raise RuntimeError(f"Failed to validate availability for connection {connection}")

def stop_db(n_nodes, port, udsport):
    """Stop a redis cluster and clear the files
    associated with it
    """
    is_uds = udsport is not None
    if is_uds:
        n_nodes = 1
    cicd = os.getenv('SR_CICD_EXECUTION')
    is_cicd = False if cicd is None else cicd.lower() == "true"

    # It's clobberin' time!
    if is_cicd:
        rediscli = 'redis-cli'
    else:
        rediscli = (
            pathlib.Path(__file__).parent.parent
            / "third-party/redis/src/redis-cli"
        ).resolve()

    # Clobber the server(s)
    procs = []
    for i in range(n_nodes):
        connection = f"-s {udsport}" if is_uds else f"-p {port + i}"
        cmd = f"{rediscli} {connection} shutdown"
        print(cmd)
        proc = Popen(cmd.split(), shell=False)
        procs.append(proc)

    # Make sure that all servers are down
    # Let exceptions propagate to the caller
    for proc in procs:
        _ = proc.communicate(timeout=15)
        if proc.returncode != 0:
            raise RuntimeError("Failed to kill Redis server!")

    # clean up after ourselves
    for i in range(n_nodes):
        fname = f"{port+i}.log"
        if os.path.exists(fname):
            os.remove(fname)

        fname = f"{port+i}.conf"
        if os.path.exists(fname):
            os.remove(fname)

    other_files = [
        'dump.rdb',
        'single.log',
        'UDS.log',
    ]
    for fname in other_files:
        if os.path.exists(fname):
            os.remove(fname)

    # Pause to give Redis time to die
    sleep(2)

def prepare_uds_socket(udsport):
    """Sets up the UDS socket"""
    if udsport is None:
        return # Silently bail
    uds_abs = pathlib.Path(udsport).resolve()
    basedir = uds_abs.parent
    basedir.mkdir(exist_ok=True)
    uds_abs.touch()
    uds_abs.chmod(0o777)

def create_db(n_nodes, port, device, rai_ver, udsport):
    """Creates a redis database starting with port at 127.0.0.1

    For a standalone server, the command issued should be equivalent to:
        redis-server --port $PORT --daemonize yes \
                     --logfile "single.log" \
                     --loadmodule $REDISAI_MODULES

    For a clustered server, the command issued should be equivalent to:
        redis-server --port $port --cluster-enabled yes --daemonize yes \
                        --cluster-config-file "$port.conf" --protected-mode no --save "" \
                        --logfile "$port.log" \
                        --loadmodule $REDISAI_MODULES

    For a UDS server, the command issued should be equivalent to:
        redis-server --unixsocket $SOCKET --unixsocketperm 777 --port 0 --bind 127.0.0.1 \
                    --daemonize yes --protected-mode no --logfile "uds.log" \
                    --loadmodule $REDISAI_MODULES

    where:
        PORT ranges from port to port + n_nodes - 1
        REDISAI_MODULES is read from the environment or calculated relative to this file
    """

    # Set up configuration
    is_uds = udsport is not None
    if is_uds:
        n_nodes = 1
    is_cluster = n_nodes > 1
    cicd = os.getenv('SR_CICD_EXECUTION')
    is_cicd = False if cicd is None else cicd.lower() == "true"

    if is_cicd:
        redisserver = "redis-server"
    else:
        redisserver = (
            pathlib.Path(__file__).parent.parent
            / "third-party/redis/src/redis-server"
        ).resolve()
    rediscli = "redis-cli" if is_cicd else os.path.dirname(redisserver) + "/redis-cli"
    test_device = device if device is not None else os.environ.get(
        "SMARTREDIS_TEST_DEVICE","cpu").lower()
    if is_cicd:
        redisai = os.getenv(f'REDISAI_{test_device.upper()}_INSTALL_PATH') + '/redisai.so'
        redisai_modules = os.getenv("REDISAI_MODULES")
        if redisai_modules is None:
            raise RuntimeError("REDISAI_MODULES environment variable is not set!")
        rai_clause = f"--loadmodule {redisai_modules}"
    else:
        if not rai_ver:
            raise RuntimeError("RedisAI version not specified")
        redisai_dir = (
            pathlib.Path(__file__).parent.parent
            / f"third-party/RedisAI/{rai_ver}/install-{test_device}"
        ).resolve()
        redisai = redisai_dir / "redisai.so"
        tf_loc = redisai_dir / "backends/redisai_tensorflow/redisai_tensorflow.so"
        torch_loc = redisai_dir / "backends/redisai_torch/redisai_torch.so"
        rai_clause = f"--loadmodule {redisai} TF {tf_loc} TORCH {torch_loc}"
    uds_clause = ""
    if is_uds:
        prepare_uds_socket(udsport)
        uds_clause = f"--bind 127.0.0.1 --unixsocket {udsport} --unixsocketperm 777"
    daemonize_clause = "--daemonize yes"
    cluster_clause = "--cluster-enabled yes" if is_cluster else ""
    prot_clause = "--protected-mode no" if is_cluster or is_uds else ""
    save_clause = '--save ""' if is_cluster else ""

    # Start servers
    procs = []
    for i in range(n_nodes):
        l_port = port + i
        port_clause = f"--port {l_port}" if not is_uds else "--port 0"
        if is_cluster:
            log_clause = f"--logfile {l_port}.log"
            cluster_cfg_clause = f"--cluster-config-file {l_port}.conf"
        else:
            log_clause = "--logfile " + ("UDS.log" if is_uds else "single.log")
            cluster_cfg_clause = ""
        log_clause += " --loglevel notice"
        cmd = f"{redisserver} {port_clause} {daemonize_clause} {cluster_clause} " + \
              f"{cluster_cfg_clause} {log_clause} {uds_clause} {rai_clause} " + \
              f"{prot_clause} {save_clause}"

        print(cmd)
        proc = Popen(cmd.split(), shell=False)
        procs.append(proc)

    # Make sure that all servers are up
    # Let exceptions propagate to the caller
    check_availability(n_nodes, port, udsport)
    for proc in procs:
        _ = proc.communicate(timeout=15)
        if proc.returncode != 0:
            raise RuntimeError("Failed to launch Redis server!")

    # Create cluster for clustered Redis request
    if n_nodes > 1:
        cluster_str = " ".join(f"127.0.0.1:{port + i}" for i in range(n_nodes))
        cmd = f"{rediscli} --cluster create {cluster_str} --cluster-replicas 0"
        print(cmd)
        proc = run(cmd.split(), input="yes", encoding="utf-8", shell=False)
        if proc.returncode != 0:
            raise SubprocessError("Cluster could not be created!")
        sleep(2)
        print("Cluster has been setup!")
    else:
        print("Server has been setup!")
    check_availability(n_nodes, port, udsport)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6379)
    parser.add_argument('--nodes', type=int, default=3)
    parser.add_argument('--rai', type=str, default=None)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--udsport', type=str, default=None)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()

    if args.stop:
        stop_db(args.nodes, args.port, args.udsport)
    else:
        create_db(args.nodes, args.port, args.device, args.rai, args.udsport)
