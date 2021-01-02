import numpy as np

from .dataset import Dataset
from .error import RedisConnectionError, RedisReplyError
from .silcPy import PyClient
from .util import Dtypes


class Client(PyClient):
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
            super().__init__(cluster, fortran)
        except RuntimeError as e:
            raise RedisConnectionError(str(e))

    def put_tensor(self, key, data):
        """Put a tensor to a Redis database

        :param key: key for tensor for be stored at
        :type key: str
        :param data: numpy array
        :type data: np.array
        :raises RedisReplyError: if put fails
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Argument provided was not a numpy array")
        dtype = Dtypes.tensor_from_numpy(data)
        try:
            super().put_tensor(key, dtype, data)
        except RuntimeError as e:
            raise RedisReplyError(str(e), key, "put_tensor") from None

    def get_tensor(self, key):
        """Get a tensor from the database

        :param key: key to get tensor from
        :type key: str
        :raises RedisReplyError: if get fails
        :return: numpy array
        :rtype: np.array
        """
        try:
            return super().get_tensor(key)
        except RuntimeError as e:
            raise RedisReplyError(str(e), key, "get_tensor") from None

    def put_dataset(self, dataset):
        """Put a Dataset instance into the database

        All associated tensors and metadata within the Dataset
        instance will also be stored

        :param dataset: a Dataset instance
        :type dataset: Dataset
        :raises TypeError: if argument is not a Dataset
        :raises RedisReplyError: if connection fails
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("Argument to put_dataset was not of type Dataset")
        else:
            try:
                super().put_dataset(dataset)
            except RuntimeError as e:
                raise RedisReplyError(str(e), dataset.name, "put_dataset") from None

    def get_dataset(self, key):
        """Get a dataset from the database

        :param key: key the dataset is stored under
        :type key: str
        :raises RedisConnectionError: if connection fails
        :return: Dataset instance
        :rtype: Dataset
        """
        try:
            dataset = super().get_dataset(key)
            return dataset
        except RuntimeError as e:
            raise RedisReplyError(str(e), key, "get_dataset") from None
