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

import pytest
from os import path as osp
from os import getcwd, environ
from glob import glob
from subprocess import Popen, PIPE, TimeoutExpired
import time

test_gpu = environ.get("SMARTREDIS_TEST_DEVICE","cpu").lower() == "gpu"

RANKS = 1
TEST_PATH = osp.dirname(osp.abspath(__file__))

def get_test_names():
    """Obtain test names by globbing for client_test
    Add tests manually if necessary
    """
    glob_path = osp.join(TEST_PATH, "client_test*")
    test_names = glob(glob_path)
    test_names = list(filter(lambda test: test.find('gpu') == -1, test_names))
    test_names = [(pytest.param(test,
                                id=osp.basename(test))) for test in test_names]
    return test_names


@pytest.mark.parametrize("test", get_test_names())
def test_fortran_client(test, build, link):
    """This function actually runs the tests using the parameterization
    function provided in Pytest

    :param test: a path to a test to run
    :type test: str
    """
    # Build the path to the test executable from the source file name
    # . keep only the last three parts of the path: (language, basename)
    test = "/".join(test.split("/")[-2:])
    # . drop the file extension
    test = ".".join(test.split(".")[:-1])
    # . prepend the path to the built test executable
    test = f"{getcwd()}/build/{build}/tests/{link}/{test}"
    cmd = [test]
    print(f"Running test: {osp.basename(test)}")
    print(f"Test command {' '.join(cmd)}")
    execute_cmd(cmd)
    time.sleep(1)

def execute_cmd(cmd_list):
    """Execute a command """

    # spawning the subprocess and connecting to its output
    proc = Popen(
        cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE, cwd=TEST_PATH)
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

@pytest.mark.skipif(
    not test_gpu,
    reason="SMARTREDIS_TEST_DEVICE does not specify 'gpu'"
)
def test_client_multigpu_mnist():
    """
    Test setting and running a machine learning model via the Fortran client
    on an orchestrator with multiple GPUs
    """
    tester_path = osp.join(TEST_PATH, "client_test_mnist_multigpu.F90")
    test_fortran_client(tester_path)

