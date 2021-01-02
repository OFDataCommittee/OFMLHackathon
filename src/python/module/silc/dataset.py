import numpy as np

from .silcPy import PyDataset
from .util import Dtypes


class Dataset(PyDataset):
    def __init__(self, name):
        """Initialize a Dataset object

        :param name: name of dataset
        :type name: str
        """
        super().__init__(name)

    def add_tensor(self, key, data):
        """Add a named tensor to this dataset

        :param key: tensor name
        :type key: str
        :param data: tensor data
        :type data: np.array
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Argument provided was not a numpy array")
        dtype = Dtypes.tensor_from_numpy(data)
        super().add_tensor(key, data, dtype)

    def get_tensor(self, key):
        """Get a tensor from this Dataset

        :param key: name of the tensor to get
        :type key: str
        :return: a numpy array of tensor data
        :rtype: np.array
        """
        return super().get_tensor(key)
