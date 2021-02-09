import numpy as np

from silc import Client

# ----- Tests -----------------------------------------------------------

CLUSTER = True


def test_1D_put_get(mock_data):
    """Test put/get_tensor for 1D numpy arrays"""

    client = Client(None, CLUSTER)

    data = mock_data.create_data(10)
    send_get_arrays(client, data)


def test_2D_put_get(mock_data):
    """Test put/get_tensor for 2D numpy arrays"""

    client = Client(None, CLUSTER)

    data = mock_data.create_data((10, 10))
    send_get_arrays(client, data)


def test_3D_put_get(mock_data):
    """Test put/get_tensor for 3D numpy arrays"""

    client = Client(None, CLUSTER)

    data = mock_data.create_data((10, 10, 10))
    send_get_arrays(client, data)


# ------- Helper Functions -----------------------------------------------


def send_get_arrays(client, data):
    """Helper for put_get tests"""

    # put to database
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        client.put_tensor(key, array)

    # get from database
    for index, array in enumerate(data):
        key = f"array_{str(index)}"
        rarray = client.get_tensor(key)
        np.testing.assert_array_equal(
            rarray, array, "Returned array from get_tensor not equal to sent tensor"
        )
