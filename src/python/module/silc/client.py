
from .silcPy import Client
from .error import ConnectionError
from .util import Dtypes


class RAIClient:

    def __init__(self, cluster=False, fortran=False):
        # TODO allow SSDB to be passed and detect if not present
        try:
            self._client = Client(cluster, fortran)
        # TODO: find out what exception this is
        except Exception:
            raise ConnectionError("Could not connect to Redis DB")

    def put_tensor(self, key, data):
        dtype = Dtypes.tensor_from_numpy(data)
        self._client.put_tensor(key, dtype, data)

    def get_tensor(self, key):
        return self._client.get_tensor(key)
