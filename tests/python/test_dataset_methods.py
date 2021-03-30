import numpy as np

from smartredis import Dataset


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


def test_add_get_scalar(mock_data):
    """Test adding and retrieving scalars to
    a dataset and with all datatypes
    """
    dataset = Dataset("test-dataset")

    # 1D tensors of all data types
    data = mock_data.create_metadata_scalars(10)
    add_get_scalars(dataset, data)


def test_add_get_strings(mock_data):
    """Test adding and retrieving strings to
    a dataset
    """
    dataset = Dataset("test-dataset")

    # list of strings
    data = mock_data.create_metadata_strings(10)
    add_get_strings(dataset, data)

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

def add_get_scalars(dataset, data):
    """Helper for metadata tests"""

    #add to dataset
    for index, scalars in enumerate(data):
        key = f"meta_scalars_{index}"
        for scalar in scalars:
            dataset.add_meta_scalar(key, scalar)

    #get from dataset
    for index, scalars in enumerate(data):
        key = f"meta_scalars_{index}"
        rscalars = dataset.get_meta_scalars(key)
        np.testing.assert_array_equal(
            rscalars,
            scalars,
            "Returned scalars from get_meta_scalars not equal to scalars added to dataset"
        )

def add_get_strings(dataset, data):
    """Helper for metadata tests"""

    #add to dataset
    key = "test_meta_strings"
    for meta_string in data:
        dataset.add_meta_string(key, meta_string)

    #get from dataset
    rdata = dataset.get_meta_strings(key)
    assert len(data) == len(rdata)
    assert all([a == b for a, b in zip(data, rdata)])