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
    glob_path = osp.join(TEST_PATH, "build/client_test*")
    test_names = glob(glob_path)
    test_names = [(pytest.param(test,
                                id=osp.basename(test))) for test in test_names]
    return test_names

def get_run_command():
    """Get run command for specific platform"""
    if which("srun"):
        return [which("srun"), "-n", f"{RANKS}"]
    return [which("mpirun"),"-np", f"{RANKS}"]

@pytest.mark.parametrize("test", get_test_names())
def test_c_client(test, use_cluster):
    """This function actually runs the tests using the parameterization
    function provided in Pytest

    :param test: a path to a test to run
    :type test: str
    """
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

