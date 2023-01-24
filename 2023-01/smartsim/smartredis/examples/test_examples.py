# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
    glob_path_1 = osp.join(TEST_PATH, "*/*/build/example*")
    glob_path_2 = osp.join(TEST_PATH, "*/*/build/smartredis*")
    test_names = glob(glob_path_1) + glob(glob_path_2)
    test_names = list(filter(lambda test: test.find('.mod') == -1, test_names))
    return test_names

@pytest.mark.parametrize("test", get_test_names())
def test_cpp_client(test, use_cluster):
    cmd = []
    cmd.append(test)
    print(f"Running test: {osp.basename(test)}")
    print(f"Test command {' '.join(cmd)}")
    print(f"Using cluster: {use_cluster}")
    execute_cmd(cmd)
    time.sleep(1)

def find_path(executable_path):
    if(executable_path.find('/') == -1):
        return '.'

    start = 0
    while True:
        slash = executable_path.find('/', start)
        if (slash == -1):
            return executable_path[0:start]
        else:
            start = slash + 1

def execute_cmd(cmd_list):
    """Execute a command """

    # spawning the subprocess and connecting to its output
    run_path = find_path(cmd_list[0])
    print("cmd_list", cmd_list)
    print("cwd", run_path)
    proc = Popen(
        cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE, cwd=run_path)
    try:
        out, err = proc.communicate(timeout=120)
        if out:
            print("OUTPUT:", out.decode("utf-8"))
        if err:
            print("ERROR:", err.decode("utf-8"))
        assert(proc.returncode == 0)
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
