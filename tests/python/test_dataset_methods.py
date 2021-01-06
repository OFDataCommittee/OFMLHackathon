import numpy as np

from silc import Dataset


def test_add_get_tensor(mock_data):
    """Test adding and retrieving 1D tensors to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 1D tensors of all data types
    data = mock_data.create_data(10)
    add_get_arrays(dataset, data)


def test_add_get_tensor_2D(mock_data):
    """Test adding and retrieving 2D tensors to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 2D tensors of all data types
    data_2D = mock_data.create_data((10, 10))
    add_get_arrays(dataset, data_2D)


def test_add_get_tensor_3D(mock_data):
    """Test adding and retrieving 3D tensors to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 3D tensors of all datatypes
    data_3D = mock_data.create_data((10, 10, 10))
    add_get_arrays(dataset, data_3D)


# ------- Helper Functions -----------------------------------------------


def add_get_arrays(dataset, data):
    """Helper for dataset tests"""

    # add to dataset
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        dataset.add_tensor(key, array)

    # get from dataset
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        rarray = dataset.get_tensor(key)
        np.testing.assert_array_equal(
            rarray,
            array,
            "Returned array from get_tensor not equal to tensor added to dataset",
        )
