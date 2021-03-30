import numpy as np
from numbers import Number

from .smartredisPy import PyDataset
from .util import Dtypes


class Dataset(PyDataset):
    def __init__(self, name):
        """Initialize a Dataset object

        :param name: name of dataset
        :type name: str
        """
        super().__init__(name)

    def add_tensor(self, name, data):
        """Add a named tensor to this dataset

        :param name: tensor name
        :type name: str
        :param data: tensor data
        :type data: np.array
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Argument provided was not a numpy array")
        dtype = Dtypes.tensor_from_numpy(data)
        super().add_tensor(name, data, dtype)

    def get_tensor(self, name):
        """Get a tensor from this Dataset

        :param name: name of the tensor to get
        :type name: str
        :return: a numpy array of tensor data
        :rtype: np.array
        """
        return super().get_tensor(name)

    def add_meta_scalar(self, name, data):
        """ Add metadata scalar field (non-string)
            with value to the DataSet.  If the
            field does not exist, it will be created.
            If the field exists, the value
            will be appended to existing field.

        :param name: The name used to reference the metadata
                     field
        :type name: str             
        :param data: a scalar
        :type data: int | float
        """

        # We want to support numpy datatypes and avoid pybind ones
        data_as_array = np.asarray(data)
        if data_as_array.size > 1:
            raise TypeError("Argument provided is not a scalar")
        # We keep dtype, in case data has a non-standard python format
        dtype = Dtypes.metadata_from_numpy(data_as_array)
        super().add_meta_scalar(name, data_as_array, dtype)

    def add_meta_string(self, name, data):
        """ Add metadata string field with value
            to the DataSet.  If the field
            does not exist, it will be created.
            If the field exists the value will
            be appended to existing field.

        :param name: The name used to reference the metadata
                     field
        :type name: str
        :param data: The string to add to the field
        :type data: str
        """
        super().add_meta_string(name, data)

    def get_meta_scalars(self, name):
        """ Get the metadata scalar field values
            from the DataSet.  
        :param name: The name used to reference the metadata
                     field in the DataSet
        :type name: str
        """
        return super().get_meta_scalars(name)

    def get_meta_strings(self, name):
        """ Get the metadata scalar field values
            from the DataSet.  
        :param name: The name used to reference the metadata
                        field in the DataSet
        :type name: str
        """
        return super().get_meta_strings(name)
