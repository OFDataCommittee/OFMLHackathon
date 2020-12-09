import numpy as np

from silc import RAIClient

# ----- Tests -----------------------------------------------------------


def test_1D_put_get():
    """Test put/get_tensor for 1D numpy arrays"""

    client = RAIClient(False, False)
    data = create_data(10)
    send_get_arrays(client, data)


def test_2D_put_get():
    """Test put/get_tensor for 2D numpy arrays"""

    client = RAIClient(False, False)
    data = create_data((10, 10))
    send_get_arrays(client, data)


def test_3D_put_get():
    """Test put/get_tensor for 3D numpy arrays"""

    client = RAIClient(False, False)
    data = create_data((10, 10, 10))
    send_get_arrays(client, data)


# ------- Helper Functions -----------------------------------------------

dtypes = [
    np.float64,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
]


def create_data(shape):
    """Helper for creating numpy data"""

    data = []
    for dtype in dtypes:
        array = np.random.randint(-10, 10, size=shape).astype(dtype)
        data.append(array)
    return data


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
