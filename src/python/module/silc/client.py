import inspect
import os
import os.path as osp

import numpy as np

from .dataset import Dataset
from .error import RedisConnectionError, RedisReplyError
from .silcPy import PyClient
from .util import Dtypes, init_default


class Client(PyClient):
    def __init__(self, address=None, cluster=False):
        """Initialize a RedisAI client.

        For clusters, the address can be a single tcp/ip address and port
        of a database node. The rest of the cluster will be discovered
        by the client itself. (e.g. address="127.0.0.1:6379")

        If an address is not set, the client will look for the environment
        variable ``$SSDB`` (e.g. SSDB="127.0.0.1:6379;")

        :param address: Address of the database
        :param cluster: True if connecting to a redis cluster, defaults to False
        :type cluster: bool, optional
        :raises RedisConnectionError: if connection initialization fails
        """
        if address:
            self.__set_address(address)
        if "SSDB" not in os.environ:
            raise RedisConnectionError()
        try:
            super().__init__(cluster)
        except RuntimeError as e:
            raise RedisConnectionError(str(e)) from None

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
            raise RedisReplyError(str(e), "put_tensor") from None

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
            raise RedisReplyError(str(e), "get_tensor") from None

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
        try:
            super().put_dataset(dataset)
        except RuntimeError as e:
            raise RedisReplyError(str(e), "put_dataset") from None

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
            raise RedisReplyError(str(e), "get_dataset", key=key) from None

    def set_function(self, key, function, device="CPU"):
        """Set a callable function into the database

        Function must be a callable TorchScript function and have at least
        one input and one output. Call the function with the Client.run_script
        method.
        Device selection is either "GPU" or "CPU". If many devices are

        present, a number can be passed for specification e.g. "GPU:1"

        :param key: key to store function at
        :type key: str
        :param function: callable function
        :type function: callable
        :param device: device to run function on, defaults to "CPU"
        :type device: str, optional
        :raises TypeError: if argument was not a callable function
        :raises RedisReplyError: if function failed to set
        """
        device = self.__check_device(device)
        if not callable(function):
            raise TypeError("Argument provided was not a callable function")
        fn_src = inspect.getsource(function)
        try:
            super().set_script(key, device, fn_src)
        except RuntimeError as e:
            raise RedisReplyError(str(e), "set_function") from None

    def set_script(self, key, script, device="CPU"):
        """Store a TorchScript at key in database

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1"

        :param key: key to store script under
        :type key: str
        :param script: TorchScript code
        :type script: str
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :raises RedisReplyError: if script fails to set
        """
        device = self.__check_device(device)
        try:
            super().set_script(key, device, script)
        except RuntimeError as e:
            raise RedisReplyError(str(e), "set_script") from None

    def set_script_from_file(self, key, file, device="CPU"):
        """Same as Client.set_script but from file

        :param key: key to store script under
        :type key: str
        :param file: path to TorchScript code
        :type file: str
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :raises RedisReplyError: if script fails to set
        """
        device = self.__check_device(device)
        file_path = self.__check_file(file)
        try:
            super().set_script_from_file(key, device, file_path)
        except RuntimeError as e:
            raise RedisReplyError(str(e), "set_script_from_file") from None

    def get_script(self, key):
        """Get a Torchscript stored in the database

        :param key: key at which script is stored
        :type key: str
        :raises RedisReplyError: if script doesn't exist
        :return: TorchScript stored at key
        :rtype: str
        """
        try:
            script = super().get_script(key)
            return script
        except RuntimeError as e:
            raise RedisReplyError(str(e), "get_script") from None

    def run_script(self, key, fn_name, inputs, outputs):
        """Execute TorchScript stored inside the database remotely

        :param key: key script is stored under
        :type key: str
        :param fn_name: name of the function within the script to execute
        :type fn_name: str
        :param inputs: list of input tensors stored in database
        :type inputs: list[str]
        :param outputs: list of output tensor names to store results under
        :type outputs: list[str]
        :raises RedisReplyError: if script execution fails
        """
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        try:
            super().run_script(key, fn_name, inputs, outputs)
        except RuntimeError as e:
            raise RedisReplyError(str(e), "run_script") from None

    def get_model(self, key):
        """Get a stored model

        :param key: key of stored model
        :type key: str
        :raises RedisReplyError: if get fails or model doesnt exist
        :return: model
        :rtype: bytes
        """
        try:
            model = super().get_model(key)
            return model
        except RuntimeError as e:
            raise RedisReplyError(str(e), "get_model")

    def set_model(
        self,
        key,
        model,
        backend,
        device="CPU",
        batch_size=0,
        min_batch_size=0,
        tag="",
        inputs=None,
        outputs=None,
    ):
        """Put a TF, TF-lite, PT, or ONNX model in the database

        :param key: key to store model under
        :type key: str
        :param model: serialized model
        :type model: bytes
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :type backend: str
        :param device: name of device for execution, defaults to "CPU"
        :type device: str, optional
        :param batch_size: batch size for execution, defaults to 0
        :type batch_size: int, optional
        :param min_batch_size: minimum batch size for model execution, defaults to 0
        :type min_batch_size: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: list[str], optional
        :param outputs: model outupts (TF only), defaults to None
        :type outputs: list[str], optional
        :raises RedisReplyError: if model fails to set
        """
        device = self.__check_device(device)
        backend = self.__check_backend(backend)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        try:
            super().set_model(
                key,
                model,
                backend,
                device,
                batch_size,
                min_batch_size,
                tag,
                inputs,
                outputs,
            )
        except RuntimeError as e:
            raise RedisReplyError(str(e), "set_model") from None

    def set_model_from_file(
        self,
        key,
        model_file,
        backend,
        device="CPU",
        batch_size=0,
        min_batch_size=0,
        tag="",
        inputs=None,
        outputs=None,
    ):
        """Put a TF, TF-lite, PT, or ONNX model from file in the database

        :param key: key to store model under
        :type key: str
        :param model_file: serialized model
        :type model_file: file path to model
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :type backend: str
        :param device: name of device for execution, defaults to "CPU"
        :type device: str, optional
        :param batch_size: batch size for execution, defaults to 0
        :type batch_size: int, optional
        :param min_batch_size: minimum batch size for model execution, defaults to 0
        :type min_batch_size: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: list[str], optional
        :param outputs: model outupts (TF only), defaults to None
        :type outputs: list[str], optional
        :raises RedisReplyError: if model fails to set
        """
        device = self.__check_device(device)
        backend = self.__check_backend(backend)
        m_file = self.__check_file(model_file)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        try:
            super().set_model_from_file(
                key,
                m_file,
                backend,
                device,
                batch_size,
                min_batch_size,
                tag,
                inputs,
                outputs,
            )
        except RuntimeError as e:
            raise RedisReplyError(str(e), "set_model_from_file") from None

    def run_model(self, key, inputs=None, outputs=None):
        """Execute a stored model

        :param key: key for stored model
        :type key: str
        :param inputs: keys of stored inputs to provide model, defaults to None
        :type inputs: list[str], optional
        :param outputs: keys to store outputs under, defaults to None
        :type outputs: list[str], optional
        :raises RedisReplyError: if model execution fails
        """
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        try:
            super().run_model(key, inputs, outputs)
        except RuntimeError as e:
            raise RedisReplyError(str(e), "run_model")

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def __check_tensor_args(inputs, outputs):
        inputs = init_default([], inputs, (list, str))
        outputs = init_default([], outputs, (list, str))
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        return inputs, outputs

    @staticmethod
    def __check_backend(backend):
        backend = backend.upper()
        if backend in ["TF", "TFLITE", "TORCH", "ONNX"]:
            return backend
        else:
            raise TypeError(f"Backend type {backend} unsupported")

    @staticmethod
    def __check_file(file):
        file_path = osp.abspath(file)
        if not osp.isfile(file_path):
            raise FileNotFoundError(file_path)
        return file_path

    @staticmethod
    def __check_device(device):
        device = device.upper()
        if not device.startswith("CPU") and not device.startswith("GPU"):
            raise TypeError("Device argument must start with either CPU or GPU")
        return device

    @staticmethod
    def __set_address(address):
        if "SSDB" in os.environ:
            del os.environ["SSDB"]
        os.environ["SSDB"] = address
