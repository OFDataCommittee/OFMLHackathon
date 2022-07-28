
from os import environ
from os.path import join
from torch import DoubleTensor


DEFAULT_DTYPE = DoubleTensor
EPS = 1.0e-6
BASE_PATH = environ.get("OFML_DRL", "")
TESTCASE_PATH = join(BASE_PATH, "test_cases")