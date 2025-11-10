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
from os import getcwd
from glob import glob
from subprocess import Popen, PIPE, TimeoutExpired
import time

RANKS = 1
TEST_PATH = osp.dirname(osp.abspath(__file__))
timeout_limit = 180

def get_test_names():
    """Obtain test names by globbing for client_test
    Add tests manually if necessary
    """
    test_names = [osp.join(TEST_PATH, "cpp_unit_tests")]
    test_names = [(pytest.param(test,
                               id=osp.basename(test))) for test in test_names]
    return test_names


@pytest.mark.parametrize("test", get_test_names())
def test_unit_cpp_client(test, build, link):
    # Build the path to the test executable from the source file name
    # . keep only the last three parts of the path: (language, unit-tests, basename)
    test = "/".join(test.split("/")[-3:])
    # . prepend the path to the built test executable
    test = f"{getcwd()}/build/{build}/tests/{link}/{test}"
    cmd = [test]
    print(f"\nRunning test: {osp.basename(test)}")
    execute_cmd(cmd)
    time.sleep(1)

def execute_cmd(cmd_list):
    """Execute a command """

    # spawning the subprocess and connecting to its output
    proc = Popen(
        cmd_list, stderr=PIPE, stdout=PIPE, stdin=PIPE, cwd=TEST_PATH)
    try:
        out, err = proc.communicate(timeout=timeout_limit)
        print("OUTPUT:", out.decode("utf-8") if out else "None")
        print("ERROR:", err.decode("utf-8") if err else "None")
        if (proc.returncode != 0):
            print("Return code:", proc.returncode)
        assert(proc.returncode == 0)
    except UnicodeDecodeError:
        output, errs = proc.communicate()
        print("OUTPUT:", out)
        print("ERROR:", errs)
        assert(False)
    except TimeoutExpired:
        proc.kill()
        output, errs = proc.communicate()
        print(f"TIMEOUT: test timed out after test timeout limit of {timeout_limit} seconds")
        print("OUTPUT:", output.decode("utf-8"))
        print("ERROR:", errs.decode("utf-8"))
        print("OUTPUT:", out)
        print("ERROR:", errs)
        assert(False)
    except AssertionError:
        assert(False)
    except Exception as e:
        print(e)
        proc.kill()
        output, errs = proc.communicate()
        print("OUTPUT:", out)
        print("ERROR:", errs)
        assert(False)
