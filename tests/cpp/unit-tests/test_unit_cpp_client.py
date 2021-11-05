# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

import pytest
from os import path as osp
from glob import glob
from shutil import which
from subprocess import Popen, PIPE, TimeoutExpired
import time

RANKS = 1
TEST_PATH = osp.dirname(osp.abspath(__file__))

def get_test_names():
    """Obtain test names by globbing for client_test
    Add tests manually if necessary
    """
    glob_path = osp.join(TEST_PATH, "build/cpp_unit_tests")
    test_names = glob(glob_path)
    test_names = [(pytest.param(test,
                               id=osp.basename(test))) for test in test_names]
    print(test_names)
    #test_names = [("build/test", "unit_tests")]
    return test_names

def get_run_command():
    """Get run command for specific platform"""
    if which("aprun"):
        return [which("aprun"), "--pes", f"{RANKS}"]
    if which("srun"):
        return [which("srun"), "-n", f"{RANKS}"]
    if which("mpirun"):
        return [which("mpirun"), "-np", f"{RANKS}"]
    raise ModuleNotFoundError("mpirun is not installed (hint: install open-mpi)")

@pytest.mark.parametrize("test", get_test_names())
def test_unit_cpp_client(test, use_cluster):
    cmd = get_run_command()
    cmd.append(test)
    print(f"Running test: {osp.basename(test)}")
    print(f"Test command {' '.join(cmd)}")
    print(f"Using cluster: {use_cluster}")
    execute_cmd(cmd)
    time.sleep(2)

def execute_cmd(cmd_list):
    """Execute a command """

    # spawning the subprocess and connecting to its output
    run_path = osp.join(TEST_PATH, "build/")
    proc = Popen(
        cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE, cwd=run_path)
    try:
        out, err = proc.communicate(timeout=120)
        assert(proc.returncode == 0)
        if out:
            print("OUTPUT:", out.decode("utf-8"))
        if err:
            print("ERROR:", err.decode("utf-8"))
    except UnicodeDecodeError:
        output, errs = proc.communicate()
        print("ERROR:", errs.decode("utf-8"))
        assert(False)
    except TimeoutExpired:
        proc.kill()
        output, errs = proc.communicate()
        print("TIMEOUT: test timed out after test timeout limit of 120 seconds")
        print("OUTPUT:", output.decode("utf-8"))
        print("ERROR:", errs.decode("utf-8"))
        assert(False)
    except Exception:
        proc.kill()
        output, errs = proc.communicate()
        print("OUTPUT:", output.decode("utf-8"))
        print("ERROR:", errs.decode("utf-8"))
        assert(False)
