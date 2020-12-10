import numpy as np

from .util import Dtypes
from .silcPy import Client
from .error import RedisConnectionError


class RAIClient:
    def __init__(self, cluster=False, fortran=False):
        """Initialize a RedisAI client.

        :param cluster: True if connecting to a redis cluster, defaults to False
        :type cluster: bool, optional
        :param fortran: True if using Fortran arrays, defaults to False
        :type fortran: bool, optional
        :raises RedisConnectionError: if connection initialization fails
        """
        # TODO allow SSDB to be passed and detect if not present
        try:
            self._client = Client(cluster, fortran)
        except RuntimeError as e:
            raise RedisConnectionError(str(e))

    def put_tensor(self, key, data):
        """Put a tensor to a Redis database

        :param key: key for tensor for be stored at
        :type key: str
        :param data: numpy array
        :type data: np.array
        :raises RedisConnectionError: if put fails
        """
        dtype = Dtypes.tensor_from_numpy(data)
        try:
            self._client.put_tensor(key, dtype, data)
        except RuntimeError as e:
            raise RedisConnectionError(str(e))

    def get_tensor(self, key):
        """Get a tensor from the database

        :param key: key to get tensor from
        :type key: str
        :raises RedisConnectionError: if get fails
        :return: numpy array
        :rtype: np.array
        """
        try:
            return self._client.get_tensor(key)
        except RuntimeError as e:
            raise RedisConnectionError(str(e))
