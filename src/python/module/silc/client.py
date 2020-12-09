from .error import RedisConnectionError
from .silcPy import Client
from .util import Dtypes


class RAIClient:
    def __init__(self, cluster=False, fortran=False):
        # TODO allow SSDB to be passed and detect if not present
        try:
            self._client = Client(cluster, fortran)
        except RuntimeError as e:
            raise RedisConnectionError(str(e))

    def put_tensor(self, key, data):
        dtype = Dtypes.tensor_from_numpy(data)
        try:
            self._client.put_tensor(key, dtype, data)
        except RuntimeError as e:
            raise RedisConnectionError(str(e))

    def get_tensor(self, key):
        try:
            return self._client.get_tensor(key)
        except RuntimeError as e:
            raise RedisConnectionError(str(e))
