# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# pylint: disable=too-many-lines,too-many-public-methods
import inspect
import os
import os.path as osp
import typing as t
import numpy as np

from .dataset import Dataset
from .configoptions import ConfigOptions
from .error import RedisConnectionError
from .smartredisPy import PyClient
from .smartredisPy import RedisReplyError as PybindRedisReplyError
from .srobject import SRObject
from .util import Dtypes, exception_handler, init_default, typecheck


class Client(SRObject):
    def __init__(self, *a: t.Any, **kw: t.Any):
        """Initialize a SmartRedis client

        At this time, the Client can be initialized with one of two
        signatures. The first version is preferred, though the second is
        supported (primarily for use in driver scripts). Note that the
        order was swapped for first two parameters in the second signature
        relative to previous releases of SmartRedis; this was necessary to
        remove ambiguity.

            Client(config_options: ConfigOptions=None,
                   logger_name: str="Default")
            Client(cluster: bool, address: optional(str)=None,
                   logger_name: str="Default")

        For detailed information on the first signature, please refer
        to the __standard_construction() method below.

        For detailed information on the second signature, please refer
        to the __address_construction() method below.

        :param a: The positional arguments supplied to this method;
                  see above for valid options
        :type a: tuple[any]; see above for valid options
        :param kw: Keyword arguments supplied to this method;
                   see above for valid options
        :type kw: dict[string, any]; see above for valid options
        :raises RedisConnectionError: if connection initialization fails
        """
        if a:
            if isinstance(a[0], bool):
                for arg in kw:
                    if arg not in ["cluster", "address", "logger_name"]:
                        raise TypeError(
                            f"__init__() got an unexpected keyword argument '{arg}'"
                        )
                pyclient = self.__address_construction(*a, **kw)
            elif isinstance(a[0], ConfigOptions) or a[0] is None:
                pyclient = self.__standard_construction(*a, **kw)
            else:
                raise TypeError(f"Invalid type for argument 0: {type(a[0])}")
        else:
            # Only kwargs in the call
            if "address" in kw or "cluster" in kw:
                pyclient = self.__address_construction(*a, **kw)
            else:
                pyclient = self.__standard_construction(*a, **kw)
        super().__init__(pyclient)

    def __address_construction(
        self,
        cluster: bool,
        address: t.Optional[str] = None,
        logger_name: str = "Default"
    ) -> PyClient:
        """Initialize a SmartRedis client

        This construction method is primarily intended for use by driver
        scripts. It is preferred to set up configuration via environment
        variables.

        For clusters, the address can be a single tcp/ip address and port
        of a database node. The rest of the cluster will be discovered
        by the client itself. (e.g. address="127.0.0.1:6379")

        If an address is not set, the client will look for the environment
        variable ``SSDB`` (e.g. SSDB="127.0.0.1:6379;")

        :param cluster: True if connecting to a redis cluster, defaults to False
        :type cluster: bool
        :param address: Address of the database
        :type address: str, optional
        :param logger_name: Identifier for the current client
        :type logger_name: str
        :raises RedisConnectionError: if connection initialization fails
        """
        if address:
            self.__set_address(address)
        if "SSDB" not in os.environ:
            raise RedisConnectionError("Could not connect to database. $SSDB not set")
        try:
            return PyClient(cluster, logger_name)
        except (PybindRedisReplyError, RuntimeError) as e:
            raise RedisConnectionError(str(e)) from None

    @staticmethod
    def __standard_construction(
        config_options: t.Optional[ConfigOptions] = None,
        logger_name: str = "Default"
    ) -> PyClient:
        """Initialize a RedisAI client

        The address of the Redis database is expected to be found in the
        SSDB environment variable (or a suffixed variable if a suffix was
        used when building the config_options object).

        :param config_options: Source for configuration data
        :type config_options: ConfigOptions, optional
        :param logger_name: Identifier for the current client
        :type logger_name: str
        :raises RedisConnectionError: if connection initialization fails
        """
        try:
            if config_options:
                pybind_config_options = config_options.get_data()
                return PyClient(pybind_config_options, logger_name)
            return PyClient(logger_name)
        except PybindRedisReplyError as e:
            raise RedisConnectionError(str(e)) from None
        except RuntimeError as e:
            raise RedisConnectionError(str(e)) from None

    def __str__(self) -> str:
        """Create a string representation of the client

        :return: A string representation of the client
        :rtype: str
        """
        return self._client.to_string()

    @property
    def _client(self) -> PyClient:
        """Alias _srobject to _client"""
        return self._srobject

    @exception_handler
    def put_tensor(self, name: str, data: np.ndarray) -> None:
        """Put a tensor to a Redis database

        The final tensor key under which the tensor is stored
        may be formed by applying a prefix to the supplied
        name. See use_tensor_ensemble_prefix() for more details.

        :param name: name for tensor for be stored at
        :type name: str
        :param data: numpy array of tensor data
        :type data: np.array
        :raises RedisReplyError: if put fails
        """
        typecheck(name, "name", str)
        typecheck(data, "data", np.ndarray)
        dtype = Dtypes.tensor_from_numpy(data)
        self._client.put_tensor(name, dtype, data)

    @exception_handler
    def get_tensor(self, name: str) -> np.ndarray:
        """Get a tensor from the database

        The tensor key used to locate the tensor
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param name: name to get tensor from
        :type name: str
        :raises RedisReplyError: if get fails
        :return: numpy array of tensor data
        :rtype: np.array
        """
        typecheck(name, "name", str)
        return self._client.get_tensor(name)

    @exception_handler
    def delete_tensor(self, name: str) -> None:
        """Delete a tensor from the database

        The tensor key used to locate the tensor to be deleted
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param name: name tensor is stored at
        :type name: str
        :raises RedisReplyError: if deletion fails
        """
        typecheck(name, "name", str)
        self._client.delete_tensor(name)

    @exception_handler
    def copy_tensor(self, src_name: str, dest_name: str) -> None:
        """Copy a tensor at one name to another name

        The source and destination tensor keys used to locate
        and store the tensor may be formed by applying prefixes
        to the supplied src_name and dest_name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param src_name: source name of tensor to be copied
        :type src_name: str
        :param dest_name: name to store new copy at
        :type dest_name: str
        :raises RedisReplyError: if copy operation fails
        """
        typecheck(src_name, "src_name", str)
        typecheck(dest_name, "dest_name", str)
        self._client.copy_tensor(src_name, dest_name)

    @exception_handler
    def rename_tensor(self, old_name: str, new_name: str) -> None:
        """Rename a tensor in the database

        The old and new tensor keys used to find and relocate
        the tensor may be formed by applying prefixes to the supplied
        old_name and new_name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param old_name: original name of tensor to be renamed
        :type old_name: str
        :param new_name: new name for the tensor
        :type new_name: str
        :raises RedisReplyError: if rename operation fails
        """
        typecheck(old_name, "old_name", str)
        typecheck(new_name, "new_name", str)
        self._client.rename_tensor(old_name, new_name)

    @exception_handler
    def put_dataset(self, dataset: Dataset) -> None:
        """Put a Dataset instance into the database

        The final dataset key under which the dataset is stored
        is generated from the name that was supplied when the
        dataset was created and may be prefixed. See
        use_dataset_ensemble_prefix() for more details.

        All associated tensors and metadata within the Dataset
        instance will also be stored.

        :param dataset: a Dataset instance
        :type dataset: Dataset
        :raises TypeError: if argument is not a Dataset
        :raises RedisReplyError: if update fails
        """
        typecheck(dataset, "dataset", Dataset)
        pybind_dataset = dataset.get_data()
        self._client.put_dataset(pybind_dataset)

    @exception_handler
    def get_dataset(self, name: str) -> Dataset:
        """Get a dataset from the database

        The dataset key used to locate the dataset
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_dataset_ensemble_prefix() for more details.

        :param name: name the dataset is stored under
        :type name: str
        :raises RedisReplyError: if retrieval fails
        :return: Dataset instance
        :rtype: Dataset
        """
        typecheck(name, "name", str)
        dataset = self._client.get_dataset(name)
        python_dataset = Dataset.from_pybind(dataset)
        return python_dataset

    @exception_handler
    def delete_dataset(self, name: str) -> None:
        """Delete a dataset within the database

        The dataset key used to locate the dataset to be deleted
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_dataset_ensemble_prefix() for more details.

        :param name: name of the dataset
        :type name: str
        :raises RedisReplyError: if deletion fails
        """
        typecheck(name, "name", str)
        self._client.delete_dataset(name)

    @exception_handler
    def copy_dataset(self, src_name: str, dest_name: str) -> None:
        """Copy a dataset from one key to another

        The source and destination dataset keys used to
        locate the dataset may be formed by applying prefixes
        to the supplied src_name and dest_name. See set_data_source()
        and use_dataset_ensemble_prefix() for more details.

        :param src_name: source name for dataset to be copied
        :type src_name: str
        :param dest_name: new name of dataset
        :type dest_name: str
        :raises RedisReplyError: if copy operation fails
        """
        typecheck(src_name, "src_name", str)
        typecheck(dest_name, "dest_name", str)
        self._client.copy_dataset(src_name, dest_name)

    @exception_handler
    def rename_dataset(self, old_name: str, new_name: str) -> None:
        """Rename a dataset in the database

        The old and new dataset keys used to find and relocate
        the dataset may be formed by applying prefixes to the supplied
        old_name and new_name. See set_data_source()
        and use_dataset_ensemble_prefix() for more details.

        :param old_name: original name of the dataset to be renamed
        :type old_name: str
        :param new_name: new name for the dataset
        :type new_name: str
        :raises RedisReplyError: if rename operation fails
        """
        typecheck(old_name, "old_name", str)
        typecheck(new_name, "new_name", str)
        self._client.rename_dataset(old_name, new_name)

    @exception_handler
    def set_function(
        self, name: str, function: t.Callable, device: str = "CPU"
    ) -> None:
        """Set a callable function into the database

        The final script key used to store the function may be formed
        by applying a prefix to the supplied name.
        See use_model_ensemble_prefix() for more details.

        Function must be a callable TorchScript function and have at least
        one input and one output. Call the function with the Client.run_script
        method.
        Device selection is either "GPU" or "CPU". If many GPUs are present,
        a zero-based index can be passed for specification e.g. "GPU:1".

        :param name: name to store function at
        :type name: str
        :param function: callable function
        :type function: callable
        :param device: device to run function on, defaults to "CPU"
        :type device: str, optional
        :raises TypeError: if argument was not a callable function
        :raises RedisReplyError: if function failed to set
        """
        typecheck(name, "name", str)
        typecheck(device, "device", str)
        if not callable(function):
            raise TypeError(
                f"Argument provided for function, {type(function)}, is not callable"
            )
        device = self.__check_device(device)
        fn_src = inspect.getsource(function)
        self._client.set_script(name, device, fn_src)

    @exception_handler
    def set_function_multigpu(
        self, name: str, function: t.Callable, first_gpu: int, num_gpus: int
    ) -> None:
        """Set a callable function into the database for use
        in a multi-GPU system

        The final script key used to store the function may be formed
        by applying a prefix to the supplied name.
        See use_model_ensemble_prefix() for more details.

        Function must be a callable TorchScript function and have at least
        one input and one output. Call the function with the Client.run_script
        method.

        :param name: name to store function at
        :type name: str
        :param function: callable function
        :type function: callable
        :param first_gpu: the first GPU (zero-based) to use in processing this function
        :type first_gpu: int
        :param num_gpus: the number of GPUs to use for this function
        :type num_gpus: int
        :raises TypeError: if argument was not a callable function
        :raises RedisReplyError: if function failed to set
        """
        typecheck(name, "name", str)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        if not callable(function):
            raise TypeError(
                f"Argument provided for function, {type(function)}, is not callable"
            )
        fn_src = inspect.getsource(function)
        self._client.set_script_multigpu(name, fn_src, first_gpu, num_gpus)

    @exception_handler
    def set_script(self, name: str, script: str, device: str = "CPU") -> None:
        """Store a TorchScript at a key in the database

        The final script key used to store the script may be formed
        by applying a prefix to the supplied name.
        See use_model_ensemble_prefix() for more details.

        Device selection is either "GPU" or "CPU". If many GPUs are present,
        a zero-based index can be passed for specification e.g. "GPU:1".

        :param name: name to store the script under
        :type name: str
        :param script: TorchScript code
        :type script: str
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :raises RedisReplyError: if script fails to set
        """
        typecheck(name, "name", str)
        typecheck(script, "script", str)
        typecheck(device, "device", str)
        device = self.__check_device(device)
        self._client.set_script(name, device, script)

    @exception_handler
    def set_script_multigpu(
        self, name: str, script: str, first_gpu: int, num_gpus: int
    ) -> None:
        """Store a TorchScript at a key in the database

        The final script key used to store the script may be formed
        by applying a prefix to the supplied name.
        See use_model_ensemble_prefix() for more details.

        :param name: name to store the script under
        :type name: str
        :param script: TorchScript code
        :type script: str
        :param first_gpu: the first GPU (zero-based) to use in processing this script
        :type first_gpu: int
        :param num_gpus: the number of GPUs to use in processing this script
        :type num_gpus: int
        :raises RedisReplyError: if script fails to set
        """
        typecheck(name, "name", str)
        typecheck(script, "script", str)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        self._client.set_script_multigpu(name, script, first_gpu, num_gpus)

    @exception_handler
    def set_script_from_file(self, name: str, file: str, device: str = "CPU") -> None:
        """Same as Client.set_script, but from file

        The final script key used to store the script may be formed
        by applying a prefix to the supplied name.
        See use_model_ensemble_prefix() for more details.

        :param name: key to store script under
        :type name: str
        :param file: path to text file containing TorchScript code
        :type file: str
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :raises RedisReplyError: if script fails to set
        """
        typecheck(name, "name", str)
        typecheck(file, "file", str)
        typecheck(device, "device", str)
        device = self.__check_device(device)
        file_path = self.__check_file(file)
        self._client.set_script_from_file(name, device, file_path)

    @exception_handler
    def set_script_from_file_multigpu(
        self, name: str, file: str, first_gpu: int, num_gpus: int
    ) -> None:
        """Same as Client.set_script_multigpu, but from file

        The final script key used to store the script may be formed
        by applying a prefix to the supplied name.
        See use_model_ensemble_prefix() for more details.

        :param name: key to store script under
        :type name: str
        :param file: path to text file containing TorchScript code
        :type file: str
        :param first_gpu: the first GPU (zero-based) to use in processing this script
        :type first_gpu: int
        :param num_gpus: the number of GPUs to use in processing this script
        :type num_gpus: int
        :raises RedisReplyError: if script fails to set
        """
        typecheck(name, "name", str)
        typecheck(file, "file", str)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        file_path = self.__check_file(file)
        self._client.set_script_from_file_multigpu(name, file_path, first_gpu, num_gpus)

    @exception_handler
    def get_script(self, name: str) -> str:
        """Retrieve a Torchscript stored in the database

        The script key used to locate the script
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and
        use_model_ensemble_prefix() for more details.

        :param name: the name at which script is stored
        :type name: str
        :raises RedisReplyError: if script retrieval fails
        :return: TorchScript stored at name
        :rtype: str
        """
        typecheck(name, "name", str)
        script = self._client.get_script(name)
        return script

    @exception_handler
    def run_script(
        self,
        name: str,
        fn_name: str,
        inputs: t.Union[str, t.List[str]],
        outputs: t.Union[str, t.List[str]]
    ) -> None:
        """Execute TorchScript stored inside the database

        The script key used to locate the script to be run
        may be formed by applying a prefix to the supplied
        name. Similarly, the tensor names in the
        input and output lists may be prefixed. See
        set_data_source(), use_model_ensemble_prefix(), and
        use_tensor_ensemble_prefix() for more details

        :param name: the name the script is stored under
        :type name: str
        :param fn_name: name of a function within the script to execute
        :type fn_name: str
        :param inputs: database tensor names to use as script inputs
        :type inputs: str | list[str]
        :param outputs: database tensor names to receive script outputs
        :type outputs: str | list[str]
        :raises RedisReplyError: if script execution fails
        """
        typecheck(name, "name", str)
        typecheck(fn_name, "fn_name", str)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.run_script(name, fn_name, inputs, outputs)

    @exception_handler
    def run_script_multigpu(
        self,
        name: str,
        fn_name: str,
        inputs: t.Union[str, t.List[str]],
        outputs: t.Union[str, t.List[str]],
        offset: int,
        first_gpu: int,
        num_gpus: int,
    ) -> None:
        """Execute TorchScript stored inside the database

        The script key used to locate the script to be run
        may be formed by applying a prefix to the supplied
        name. Similarly, the tensor names in the
        input and output lists may be prefixed. See
        set_data_source(), use_model_ensemble_prefix(), and
        use_tensor_ensemble_prefix() for more details

        :param name: the name the script is stored under
        :type name: str
        :param fn_name: name of a function within the script to execute
        :type fn_name: str
        :param inputs: database tensor names to use as script inputs
        :type inputs: str | list[str]
        :param outputs: database tensor names to receive script outputs
        :type outputs: str | list[str]
        :param offset: index of the current image, such as a processor ID
                         or MPI rank
        :type offset: int
        :param first_gpu: the first GPU (zero-based) to use in processing this script
        :type first_gpu: int
        :param num_gpus: the number of gpus for which the script was stored
        :type num_gpus: int
        :raises RedisReplyError: if script execution fails
        """
        typecheck(name, "name", str)
        typecheck(fn_name, "fn_name", str)
        typecheck(offset, "offset", int)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.run_script_multigpu(
            name, fn_name, inputs, outputs, offset, first_gpu, num_gpus
        )

    @exception_handler
    def delete_script(self, name: str) -> None:
        """Remove a script from the database

        The script key used to locate the script to be run
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_model_ensemble_prefix()
        for more details

        :param name: the name the script is stored under
        :type name: str
        :raises RedisReplyError: if script deletion fails
        """
        typecheck(name, "name", str)
        self._client.delete_script(name)

    @exception_handler
    def delete_script_multigpu(self, name: str, first_gpu: int, num_gpus: int) -> None:
        """Remove a script from the database

        The script key used to locate the script to be run
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_model_ensemble_prefix()
        for more details

        :param name: the name the script is stored under
        :type name: str
        :param first_gpu: the first GPU (zero-based) to use in processing this script
        :type first_gpu: int
        :param num_gpus: the number of gpus for which the script was stored
        :type num_gpus: int
        :raises RedisReplyError: if script deletion fails
        """
        typecheck(name, "name", str)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        self._client.delete_script_multigpu(name, first_gpu, num_gpus)

    @exception_handler
    def get_model(self, name: str) -> bytes:
        """Get a stored model

        The model key used to locate the model
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_model_ensemble_prefix() for more details.

        :param name: name of stored model
        :type name: str
        :raises RedisReplyError: if retrieval fails
        :return: model
        :rtype: bytes
        """
        typecheck(name, "name", str)
        model = self._client.get_model(name)
        return model

    @exception_handler
    def set_model(
        self,
        name: str,
        model: bytes,
        backend: str,
        device: str = "CPU",
        batch_size: int = 0,
        min_batch_size: int = 0,
        min_batch_timeout: int = 0,
        tag: str = "",
        inputs: t.Optional[t.Union[str, t.List[str]]] = None,
        outputs: t.Optional[t.Union[str, t.List[str]]] = None,
    ) -> None:
        """Put a TF, TF-lite, PT, or ONNX model in the database

        The final model key used to store the model
        may be formed by applying a prefix to the supplied
        name. Similarly, the tensor names in the
        input and output nodes for TF models may be prefixed.
        See set_data_source(), use_model_ensemble_prefix(), and
        use_tensor_ensemble_prefix() for more details.
        Device selection is either "GPU" or "CPU". If many GPUs are present,
        a zero-based index can be passed for specification e.g. "GPU:1".

        :param name: name to store model under
        :type name: str
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
        :param min_batch_timeout: Max time (ms) to wait for min batch size
        :type min_batch_timeout: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: str | list[str] | None
        :param outputs: model outputs (TF only), defaults to None
        :type outputs: str | list[str] | None
        :raises RedisReplyError: if model fails to set
        """
        typecheck(name, "name", str)
        typecheck(backend, "backend", str)
        typecheck(device, "device", str)
        typecheck(batch_size, "batch_size", int)
        typecheck(min_batch_size, "min_batch_size", int)
        typecheck(min_batch_timeout, "min_batch_timeout", int)
        typecheck(tag, "tag", str)
        device = self.__check_device(device)
        backend = self.__check_backend(backend)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.set_model(
            name,
            model,
            backend,
            device,
            batch_size,
            min_batch_size,
            min_batch_timeout,
            tag,
            inputs,
            outputs,
        )

    @exception_handler
    def set_model_multigpu(
        self,
        name: str,
        model: bytes,
        backend: str,
        first_gpu: int,
        num_gpus: int,
        batch_size: int = 0,
        min_batch_size: int = 0,
        min_batch_timeout: int = 0,
        tag: str = "",
        inputs: t.Optional[t.Union[str, t.List[str]]] = None,
        outputs: t.Optional[t.Union[str, t.List[str]]] = None,
    ) -> None:
        """Put a TF, TF-lite, PT, or ONNX model in the database for use
        in a multi-GPU system

        The final model key used to store the model
        may be formed by applying a prefix to the supplied
        name. Similarly, the tensor names in the
        input and output nodes for TF models may be prefixed.
        See set_data_source(), use_model_ensemble_prefix(), and
        use_tensor_ensemble_prefix() for more details.

        :param name: name to store model under
        :type name: str
        :param model: serialized model
        :type model: bytes
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :type backend: str
        :param first_gpu: the first GPU (zero-based) to use in processing this model
        :type first_gpu: int
        :param num_gpus: the number of GPUs to use in processing this model
        :type num_gpus: int
        :param batch_size: batch size for execution, defaults to 0
        :type batch_size: int, optional
        :param min_batch_size: minimum batch size for model execution, defaults to 0
        :type min_batch_size: int, optional
        :param min_batch_timeout: Max time (ms) to wait for min batch size
        :type min_batch_timeout: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: str | list[str] | None
        :param outputs: model outputs (TF only), defaults to None
        :type outputs: str | list[str] | None
        :raises RedisReplyError: if model fails to set
        """
        typecheck(name, "name", str)
        typecheck(backend, "backend", str)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        typecheck(batch_size, "batch_size", int)
        typecheck(min_batch_size, "min_batch_size", int)
        typecheck(min_batch_timeout, "min_batch_timeout", int)
        typecheck(tag, "tag", str)
        backend = self.__check_backend(backend)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.set_model_multigpu(
            name,
            model,
            backend,
            first_gpu,
            num_gpus,
            batch_size,
            min_batch_size,
            min_batch_timeout,
            tag,
            inputs,
            outputs,
        )

    @exception_handler
    def set_model_from_file(
        self,
        name: str,
        model_file: str,
        backend: str,
        device: str = "CPU",
        batch_size: int = 0,
        min_batch_size: int = 0,
        min_batch_timeout: int = 0,
        tag: str = "",
        inputs: t.Optional[t.Union[str, t.List[str]]] = None,
        outputs: t.Optional[t.Union[str, t.List[str]]] = None,
    ) -> None:
        """Put a TF, TF-lite, PT, or ONNX model from file in the database

        The final model key used to store the model
        may be formed by applying a prefix to the supplied
        name. Similarly, the tensor names in the
        input and output nodes for TF models may be prefixed.
        See set_data_source(), use_model_ensemble_prefix(), and
        use_tensor_ensemble_prefix() for more details.
        Device selection is either "GPU" or "CPU". If many GPUs are present,
        a zero-based index can be passed for specification e.g. "GPU:1".

        :param name: name to store model under
        :type name: str
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
        :param min_batch_timeout: Max time (ms) to wait for min batch size
        :type min_batch_timeout: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: str | list[str] | None
        :param outputs: model outupts (TF only), defaults to None
        :type outputs: str | list[str] | None
        :raises RedisReplyError: if model fails to set
        """
        typecheck(name, "name", str)
        typecheck(model_file, "model_file", str)
        typecheck(backend, "backend", str)
        typecheck(device, "device", str)
        typecheck(batch_size, "batch_size", int)
        typecheck(min_batch_size, "min_batch_size", int)
        typecheck(min_batch_timeout, "min_batch_timeout", int)
        typecheck(tag, "tag", str)
        device = self.__check_device(device)
        backend = self.__check_backend(backend)
        m_file = self.__check_file(model_file)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.set_model_from_file(
            name,
            m_file,
            backend,
            device,
            batch_size,
            min_batch_size,
            min_batch_timeout,
            tag,
            inputs,
            outputs,
        )

    @exception_handler
    def set_model_from_file_multigpu(
        self,
        name: str,
        model_file: str,
        backend: str,
        first_gpu: int,
        num_gpus: int,
        batch_size: int = 0,
        min_batch_size: int = 0,
        min_batch_timeout: int = 0,
        tag: str = "",
        inputs: t.Optional[t.Union[str, t.List[str]]] = None,
        outputs: t.Optional[t.Union[str, t.List[str]]] = None,
    ) -> None:
        """Put a TF, TF-lite, PT, or ONNX model from file in the database
        for use in a multi-GPU system

        The final model key used to store the model
        may be formed by applying a prefix to the supplied
        name. Similarly, the tensor names in the
        input and output nodes for TF models may be prefixed.
        See set_data_source(), use_model_ensemble_prefix(), and
        use_tensor_ensemble_prefix() for more details.

        :param name: name to store model under
        :type name: str
        :param model_file: serialized model
        :type model_file: file path to model
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :type backend: str
        :param first_gpu: the first GPU (zero-based) to use in processing this model
        :type first_gpu: int
        :param num_gpus: the number of GPUs to use in processing this model
        :type num_gpus: int
        :param batch_size: batch size for execution, defaults to 0
        :type batch_size: int, optional
        :param min_batch_size: minimum batch size for model execution, defaults to 0
        :type min_batch_size: int, optional
        :param min_batch_timeout: Max time (ms) to wait for min batch size
        :type min_batch_timeout: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: str | list[str] | None
        :param outputs: model outupts (TF only), defaults to None
        :type outputs: str | list[str] | None
        :raises RedisReplyError: if model fails to set
        """
        typecheck(name, "name", str)
        typecheck(model_file, "model_file", str)
        typecheck(backend, "backend", str)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        typecheck(batch_size, "batch_size", int)
        typecheck(min_batch_size, "min_batch_size", int)
        typecheck(min_batch_timeout, "min_batch_timeout", int)
        typecheck(tag, "tag", str)
        backend = self.__check_backend(backend)
        m_file = self.__check_file(model_file)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.set_model_from_file_multigpu(
            name,
            m_file,
            backend,
            first_gpu,
            num_gpus,
            batch_size,
            min_batch_size,
            min_batch_timeout,
            tag,
            inputs,
            outputs,
        )

    @exception_handler
    def run_model(
        self,
        name: str,
        inputs: t.Optional[t.Union[str, t.List[str]]] = None,
        outputs: t.Optional[t.Union[str, t.List[str]]] = None,
    ) -> None:
        """Execute a stored model

        The model key used to locate the model to be run
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_model_ensemble_prefix() for more details.

        :param name: name for stored model
        :type name: str
        :param inputs: names of stored inputs to provide model, defaults to None
        :type inputs: str | list[str] | None
        :param outputs: names to store outputs under, defaults to None
        :type outputs: str | list[str] | None
        :raises RedisReplyError: if model execution fails
        """
        typecheck(name, "name", str)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.run_model(name, inputs, outputs)

    @exception_handler
    def run_model_multigpu(
        self,
        name: str,
        offset: int,
        first_gpu: int,
        num_gpus: int,
        inputs: t.Optional[t.Union[str, t.List[str]]] = None,
        outputs: t.Optional[t.Union[str, t.List[str]]] = None,
    ) -> None:
        """Execute a model stored for a multi-GPU system

        The model key used to locate the model to be run
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_model_ensemble_prefix() for more details.

        :param name: name for stored model
        :type name: str
        :param offset: index of the current image, such as a processor ID
                         or MPI rank
        :type offset: int
        :param first_gpu: the first GPU (zero-based) to use in processing this model
        :type first_gpu: int
        :param num_gpus: the number of gpus for which the model was stored
        :type num_gpus: int
        :param inputs: names of stored inputs to provide model, defaults to None
        :type inputs: str | list[str] | None
        :param outputs: names to store outputs under, defaults to None
        :type outputs: str | list[str] | None
        :raises RedisReplyError: if model execution fails
        """
        typecheck(name, "name", str)
        typecheck(offset, "offset", int)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        self._client.run_model_multigpu(
            name, inputs, outputs, offset, first_gpu, num_gpus
        )

    @exception_handler
    def delete_model(self, name: str) -> None:
        """Remove a model from the database

        The model key used to locate the script to be run
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_model_ensemble_prefix()
        for more details

        :param name: the name the model is stored under
        :type name: str
        :raises RedisReplyError: if model deletion fails
        """
        typecheck(name, "name", str)
        self._client.delete_model(name)

    @exception_handler
    def delete_model_multigpu(self, name: str, first_gpu: int, num_gpus: str) -> None:
        """Remove a model from the database that was stored for use with multiple GPUs

        The model key used to locate the script to be run
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_model_ensemble_prefix()
        for more details

        :param name: the name the model is stored under
        :type name: str
        :param first_gpu: the first GPU (zero-based) to use in processing this model
        :type first_gpu: int
        :param num_gpus: the number of gpus for which the model was stored
        :type num_gpus: int
        :raises RedisReplyError: if model deletion fails
        """
        typecheck(name, "name", str)
        typecheck(first_gpu, "first_gpu", int)
        typecheck(num_gpus, "num_gpus", int)
        self._client.delete_model_multigpu(name, first_gpu, num_gpus)

    @exception_handler
    def tensor_exists(self, name: str) -> bool:
        """Check if a tensor exists in the database

        The tensor key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param name: The tensor name that will be checked in the database
        :type name: str
        :returns: Returns true if the tensor exists in the database
        :rtype: bool
        :raises RedisReplyError: if checking for tensor existence causes an error
        """
        typecheck(name, "name", str)
        return self._client.tensor_exists(name)

    @exception_handler
    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset exists in the database

        The dataset key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_dataset_ensemble_prefix() for more details.

        :param name: The dataset name that will be checked in the database
        :type name: str
        :returns: Returns true if the dataset exists in the database
        :rtype: bool
        :raises RedisReplyError: if `dataset_exists` fails (i.e. causes an error)
        """
        typecheck(name, "name", str)
        return self._client.dataset_exists(name)

    @exception_handler
    def model_exists(self, name: str) -> bool:
        """Check if a model or script exists in the database

        The model or script key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_model_ensemble_prefix() for more details.

        :param name: The model or script name that will be checked in the database
        :type name: str
        :returns: Returns true if the model exists in the database
        :rtype: bool
        :raises RedisReplyError: if `model_exists` fails (i.e. causes an error)
        """
        typecheck(name, "name", str)
        return self._client.model_exists(name)

    @exception_handler
    def key_exists(self, key: str) -> bool:
        """Check if the key exists in the database

        :param key: The key that will be checked in the database
        :type key: str
        :returns: Returns true if the key exists in the database
        :rtype: bool
        :raises RedisReplyError: if `key_exists` fails
        """
        typecheck(key, "key", str)
        return self._client.key_exists(key)

    @exception_handler
    def poll_key(self, key: str, poll_frequency_ms: int, num_tries: int) -> bool:
        """Check if the key exists in the database

        The check is repeated at a specified polling interval and for
        a specified number of retries.

        :param key: The key that will be checked in the database
        :type key: str
        :param poll_frequency_ms: The polling interval, in milliseconds
        :type poll_frequency_ms: int
        :param num_tries: The total number of retries for the check
        :type num_tries: int
        :returns: Returns true if the key is found within the
                  specified number of tries, otherwise false.
        :rtype: bool
        :raises RedisReplyError: if an error occurs while polling
        """
        typecheck(key, "key", str)
        typecheck(poll_frequency_ms, "poll_frequency_ms", int)
        typecheck(num_tries, "num_tries", int)
        return self._client.poll_key(key, poll_frequency_ms, num_tries)

    @exception_handler
    def poll_tensor(self, name: str, poll_frequency_ms: int, num_tries: int) -> bool:
        """Check if a tensor exists in the database

        The check is repeated at a specified polling interval and for
        a specified number of retries.
        The tensor key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param name: The tensor name that will be checked in the database
        :type name: str
        :param poll_frequency_ms: The polling interval, in milliseconds
        :type poll_frequency_ms: int
        :param num_tries: The total number of retries for the check
        :type num_tries: int
        :returns: Returns true if the tensor key is found within the
                  specified number of tries, otherwise false.
        :rtype: bool
        :raises RedisReplyError: if an error occurs while polling
        """
        typecheck(name, "name", str)
        typecheck(poll_frequency_ms, "poll_frequency_ms", int)
        typecheck(num_tries, "num_tries", int)
        return self._client.poll_tensor(name, poll_frequency_ms, num_tries)

    @exception_handler
    def poll_dataset(self, name: str, poll_frequency_ms: int, num_tries: int) -> bool:
        """Check if a dataset exists in the database

        The check is repeated at a specified polling interval and for
        a specified number of retries.
        The dataset key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_dataset_ensemble_prefix() for more details.

        :param name: The dataset name that will be checked in the database
        :type name: str
        :param poll_frequency_ms: The polling interval, in milliseconds
        :type poll_frequency_ms: int
        :param num_tries: The total number of retries for the check
        :type num_tries: int
        :returns: Returns true if the key is found within the
                  specified number of tries, otherwise false.
        :rtype: bool
        :raises RedisReplyError: if an error occurs while polling
        """
        typecheck(name, "name", str)
        typecheck(poll_frequency_ms, "poll_frequency_ms", int)
        typecheck(num_tries, "num_tries", int)
        return self._client.poll_dataset(name, poll_frequency_ms, num_tries)

    @exception_handler
    def poll_model(self, name: str, poll_frequency_ms: int, num_tries: int) -> bool:
        """Check if a model or script exists in the database

        The check is repeated at a specified polling interval and for
        a specified number of retries.
        The model or script key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_model_ensemble_prefix() for more details.

        :param name: The model or script name that will be checked in the database
        :type name: str
        :param poll_frequency_ms: The polling interval, in milliseconds
        :type poll_frequency_ms: int
        :param num_tries: The total number of retries for the check
        :type num_tries: int
        :returns: Returns true if the key is found within the
                  specified number of tries, otherwise false.
        :rtype: bool
        :raises RedisReplyError: if an error occurs while polling
        """
        typecheck(name, "name", str)
        typecheck(poll_frequency_ms, "poll_frequency_ms", int)
        typecheck(num_tries, "num_tries", int)
        return self._client.poll_model(name, poll_frequency_ms, num_tries)

    @exception_handler
    def set_data_source(self, source_id: str) -> None:
        """Set the data source, a key prefix for future operations

        When running multiple applications, such as an ensemble
        computation, there is a risk that the same name is used
        for a tensor, dataset, script, or model by more than one
        executing entity. In order to prevent this sort of collision,
        SmartRedis affords the ability to add a prefix to names,
        thereby associating them with the name of the specific
        entity that the prefix corresponds to. For writes to
        the database when prefixing is activated, the prefix
        used is taken from the SSKEYOUT environment variable.
        For reads from the database, the default is to use the
        first prefix from SSKEYIN. If this is the same as the
        prefix from SSKEYOUT, the entity will read back the
        same data it wrote; however, this function allows an entity
        to read from data written by another entity (i.e. use
        the other entity's key.)

        :param source_id: The prefix for read operations; must have
                          previously been set via the SSKEYIN environment
                          variable
        :type source_id: str
        :raises RedisReplyError: if set data
        """
        typecheck(source_id, "source_id", str)
        return self._client.set_data_source(source_id)

    @exception_handler
    def use_model_ensemble_prefix(self, use_prefix: bool) -> None:
        """Control whether model and script keys are
           prefixed (e.g. in an ensemble) when forming database keys

        This function can be used to avoid key collisions in an ensemble
        by prepending the string value from the environment variable SSKEYIN
        to model and script names.
        Prefixes will only be used if they were previously set through
        environment variables SSKEYIN and SSKEYOUT.
        Keys for entities created before this function is called
        will not be retroactively prefixed.
        By default, the client does not prefix model and script
        keys.

        :param use_prefix: If set to true, all future operations
                           on models and scripts will use a prefix, if
                           available.
        :type use_prefix: bool
        """
        typecheck(use_prefix, "use_prefix", bool)
        return self._client.use_model_ensemble_prefix(use_prefix)

    @exception_handler
    def use_list_ensemble_prefix(self, use_prefix: bool) -> None:
        """Control whether aggregation lists are prefixed
           when forming database keys

        This function can be used to avoid key collisions in an
        ensemble by prepending the string value from the
        environment variable SSKEYIN and/or SSKEYOUT to
        aggregation list names.  Prefixes will only be used if
        they were previously set through the environment variables
        SSKEYOUT and SSKEYIN. Keys for aggregation lists created
        before this function is called will not be retroactively
        prefixed. By default, the client prefixes aggregation
        list keys with the first prefix specified with the SSKEYIN
        and SSKEYOUT environment variables.  Note that
        use_dataset_ensemble_prefix() controls prefixing
        for the entities in the aggregation list, and
        use_dataset_ensemble_prefix() should be given the
        same value that was used during the initial
        setting of the DataSet into the database.

        :param use_prefix: If set to true, all future operations
                           on aggregation lists will use a prefix, if
                           available.
        :type use_prefix: bool
        """
        typecheck(use_prefix, "use_prefix", bool)
        return self._client.use_list_ensemble_prefix(use_prefix)

    @exception_handler
    def use_tensor_ensemble_prefix(self, use_prefix: bool) -> None:
        """Control whether tensor keys are prefixed (e.g. in an
        ensemble) when forming database keys

        This function can be used to avoid key collisions in an ensemble
        by prepending the string value from the environment variable SSKEYIN
        to tensor names.
        Prefixes will only be used if they were previously set through
        environment variables SSKEYIN and SSKEYOUT.
        Keys for entities created before this function is called
        will not be retroactively prefixed.
        By default, the client prefixes tensor keys when a prefix is
        available.

        :param use_prefix: If set to true, all future operations on tensors
                           will use a prefix, if available.
        :type use_prefix: bool
        """
        typecheck(use_prefix, "use_prefix", bool)
        return self._client.use_tensor_ensemble_prefix(use_prefix)

    @exception_handler
    def use_dataset_ensemble_prefix(self, use_prefix: bool) -> None:
        """Control whether dataset keys are prefixed (e.g. in an ensemble)
           when forming database keys

        This function can be used to avoid key collisions in an ensemble
        by prepending the string value from the environment variable SSKEYIN
        to dataset names.
        Prefixes will only be used if they were previously set through
        environment variables SSKEYIN and SSKEYOUT.
        Keys for entities created before this function is called
        will not be retroactively prefixed.
        By default, the client prefixes dataset keys when a prefix is
        available.

        :param use_prefix: If set to true, all future operations on datasets
                           will use a prefix, if available.
        :type use_prefix: bool
        """
        typecheck(use_prefix, "use_prefix", bool)
        return self._client.use_dataset_ensemble_prefix(use_prefix)

    @exception_handler
    def get_db_node_info(self, addresses: t.List[str]) -> t.List[t.Dict]:
        """Returns information about given database nodes

        :param addresses: The addresses of the database nodes
        :type address: list[str]
        :returns: A list of dictionaries with each entry in the
                  list corresponding to an address reply
        :rtype: list[dict]
        :raises RedisReplyError: if there is an error
                in command execution or the address
                is not reachable by the client.
                In the case of using a cluster of database nodes,
                it is best practice to bind each node in the cluster
                to a specific address to avoid inconsistencies in
                addresses retrieved with the CLUSTER SLOTS command.
                Inconsistencies in node addresses across
                CLUSTER SLOTS commands can lead to RedisReplyError
                being thrown.
        """
        typecheck(addresses, "addresses", list)
        return self._client.get_db_node_info(addresses)

    @exception_handler
    def get_db_cluster_info(self, addresses: t.List[str]) -> t.List[t.Dict]:
        """Returns cluster information from a specified db node.
        If the address does not correspond to a cluster node,
        an empty dictionary is returned.

        :param addresses: The addresses of the database nodes
        :type address: list[str]
        :returns: A list of dictionaries with each entry in the
                  list corresponding to an address reply
        :rtype: list[dict]
        :raises RedisReplyError: if there is an error
                in command execution or the address
                is not reachable by the client or if on a
                non-cluster environment.
                In the case of using a cluster of database nodes,
                it is best practice to bind each node in the cluster
                to a specific address to avoid inconsistencies in
                addresses retrieved with the CLUSTER SLOTS command.
                Inconsistencies in node addresses across
                CLUSTER SLOTS commands can lead to RedisReplyError
                being thrown.
        """
        typecheck(addresses, "addresses", list)
        return self._client.get_db_cluster_info(addresses)

    @exception_handler
    def get_ai_info(
        self, address: t.List[str], key: str, reset_stat: bool = False
    ) -> t.List[t.Dict]:
        """Returns AI.INFO command reply information for the
        script or model key at the provided addresses.

        :param addresses: The addresses of the database nodes
        :type address: list[str]
        :param key: The key associated with the model or script
        :type key: str
        :param reset_stat: Boolean indicating if the statistics
                           for the model or script should be
                           reset.
        :type reset_stat: bool
        :returns: A list of dictionaries with each entry in the
                  list corresponding to an address reply
        :rtype: list[dict]
        :raises RedisReplyError: if there is an error
                in command execution or parsing the command reply.
        """
        typecheck(address, "address", list)
        typecheck(key, "key", str)
        typecheck(reset_stat, "reset_stat", bool)
        return self._client.get_ai_info(address, key, reset_stat)

    @exception_handler
    def flush_db(self, addresses: t.List[str]) -> None:
        """Removes all keys from a specified db node.

        :param addresses: The addresses of the database nodes
        :type address: list[str]
        :raises RedisReplyError: if there is an error
                in command execution or the address
                is not reachable by the client.
                In the case of using a cluster of database nodes,
                it is best practice to bind each node in the cluster
                to a specific address to avoid inconsistencies in
                addresses retrieved with the CLUSTER SLOTS command.
                Inconsistencies in node addresses across
                CLUSTER SLOTS commands can lead to RedisReplyError
                being thrown.
        """
        typecheck(addresses, "addresses", list)
        self._client.flush_db(addresses)

    @exception_handler
    def config_get(self, expression: str, address: t.List[str]) -> t.Dict:
        """Read the configuration parameters of a running server.
        If the address does not correspond to a cluster node,
        an empty dictionary is returned.

        :param expression: Parameter used in the configuration or a
                           glob pattern (Use '*' to retrieve all
                           configuration parameters)
        :type expression: str
        :param address: The address of the database node
        :type address: str
        :returns: A dictionary that maps configuration parameters to
                  their values. If the provided expression does not
                  exist, then an empty dictionary is returned.
        :rtype: dict
        :raises RedisReplyError: if there is an error
                in command execution or the address
                is not reachable by the client.
                In the case of using a cluster of database nodes,
                it is best practice to bind each node in the cluster
                to a specific address to avoid inconsistencies in
                addresses retrieved with the CLUSTER SLOTS command.
                Inconsistencies in node addresses across
                CLUSTER SLOTS commands can lead to RedisReplyError
                being thrown.
        """
        typecheck(expression, "expression", str)
        typecheck(address, "address", str)
        return self._client.config_get(expression, address)

    @exception_handler
    def config_set(self, config_param: str, value: str, address: str) -> None:
        """Reconfigure the server. It can change both trivial
        parameters or switch from one to another persistence option.
        All the configuration parameters set using this command are
        immediately loaded by Redis and will take effect starting with
        the next command executed.
        If the address does not correspond to a cluster node,
        an empty dictionary is returned.

        :param config_param: A configuration parameter to set
        :type config_param: str
        :param value: The value to assign to the configuration parameter
        :type value: str
        :param address: The address of the database node
        :type address: str
        :raises RedisReplyError: if there is an error
                in command execution or the address
                is not reachable by the client or if the config_param
                is unsupported. In the case of using a cluster of
                database nodes, it is best practice to bind each node
                in the cluster to a specific address to avoid inconsistencies
                in addresses retrieved with the CLUSTER SLOTS command.
                Inconsistencies in node addresses across
                CLUSTER SLOTS commands can lead to RedisReplyError
                being thrown.
        """
        typecheck(config_param, "config_param", str)
        typecheck(value, "value", str)
        typecheck(address, "address", str)
        self._client.config_set(config_param, value, address)

    @exception_handler
    def save(self, addresses: t.List[str]) -> None:
        """Performs a synchronous save of the database shard
        producing a point in time snapshot of all the data
        inside the Redis instance, in the form of an RBD file.

        :param addresses: The addresses of the database nodes
        :type addresses: list[str]
        :raises RedisReplyError: if there is an error
                in command execution or the address
                is not reachable by the client.
                In the case of using a cluster of database nodes,
                it is best practice to bind each node in the cluster
                to a specific address to avoid inconsistencies in
                addresses retrieved with the CLUSTER SLOTS command.
                Inconsistencies in node addresses across
                CLUSTER SLOTS commands can lead to RedisReplyError
                being thrown.
        """
        typecheck(addresses, "addresses", list)
        self._client.save(addresses)

    @exception_handler
    def set_model_chunk_size(self, chunk_size: int) -> None:
        """Reconfigures the chunking size that Redis uses for model
           serialization, replication, and the model_get command.
           This method triggers the AI.CONFIG method in the Redis
           database to change the model chunking size.

           NOTE: The default size of 511MB should be fine for most
           applications, so it is expected to be very rare that a
           client calls this method. It is not necessary to call
           this method a model to be chunked.
        :param chunk_size: The new chunk size in bytes
        :type addresses: int
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(chunk_size, "chunk_size", int)
        self._client.set_model_chunk_size(chunk_size)

    @exception_handler
    def append_to_list(self, list_name: str, dataset: Dataset) -> None:
        """Appends a dataset to the aggregation list

        When appending a dataset to an aggregation list,
        the list will automatically be created if it does not
        exist (i.e. this is the first entry in the list).
        Aggregation lists work by referencing the dataset
        by storing its key, so appending a dataset
        to an aggregation list does not create a copy of the
        dataset.  Also, for this reason, the dataset
        must have been previously placed into the database
        with a separate call to put_dataset().

        :param list_name: The name of the aggregation list
        :type list_name: str
        :param dataset: The DataSet to append
        :type dataset: Dataset
        :raises TypeError: if argument is not a Dataset
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(list_name, "list_name", str)
        typecheck(dataset, "dataset", Dataset)
        pybind_dataset = dataset.get_data()
        self._client.append_to_list(list_name, pybind_dataset)

    @exception_handler
    def delete_list(self, list_name: str) -> None:
        """Delete an aggregation list

        The key used to locate the aggregation list to be
        deleted may be formed by applying a prefix to the
        supplied name. See set_data_source()
        and use_list_ensemble_prefix() for more details.

        :param list_name: The name of the aggregation list
        :type list_name: str
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(list_name, "list_name", str)
        self._client.delete_list(list_name)

    @exception_handler
    def copy_list(self, src_name: str, dest_name: str) -> None:
        """Copy an aggregation list

        The source and destination aggregation list keys used to
        locate and store the aggregation list may be formed by
        applying prefixes to the supplied src_name and dest_name.
        See set_data_source() and use_list_ensemble_prefix()
        for more details.

        :param src_name: The source list name
        :type src_name: str
        :param dest_name: The destination list name
        :type dest_name: str
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(src_name, "src_name", str)
        typecheck(dest_name, "dest_name", str)
        self._client.copy_list(src_name, dest_name)

    @exception_handler
    def rename_list(self, src_name: str, dest_name: str) -> None:
        """Rename an aggregation list

        The old and new aggregation list key used to find and
        relocate the list may be formed by applying prefixes to
        the supplied old_name and new_name. See set_data_source()
        and use_list_ensemble_prefix() for more details.

        :param src_name: The source list name
        :type src_name: str
        :param dest_name: The destination list name
        :type dest_name: str
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(src_name, "src_name", str)
        typecheck(dest_name, "dest_name", str)
        self._client.rename_list(src_name, dest_name)

    @exception_handler
    def get_list_length(self, list_name: str) -> int:
        """Get the number of entries in the list

        :param list_name: The list name
        :type list_name: str
        :return: The length of the list
        :rtype: int
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(list_name, "list_name", str)
        return self._client.get_list_length(list_name)

    @exception_handler
    def poll_list_length(
        self, name: str, list_length: int, poll_frequency_ms: int, num_tries: int
    ) -> bool:
        """Poll list length until length is equal
        to the provided length.  If maximum number of
        attempts is exceeded, returns False

        The aggregation list key used to check for list length
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_list_ensemble_prefix()
        for more details.

        :param name: The name of the list
        :type name: str
        :param list_length: The desired length of the list
        :type list_length: int
        :param poll_frequency_ms: The time delay between checks, in milliseconds
        :type poll_frequency_ms: int
        :param num_tries: The total number of times to check for the name
        :type num_tries: int
        :return: Returns true if the list is found with a length greater
                than or equal to the provided length, otherwise false
        :rtype: bool
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(name, "name", str)
        typecheck(list_length, "list_length", int)
        typecheck(poll_frequency_ms, "poll_frequency_ms", int)
        typecheck(num_tries, "num_tries", int)
        return self._client.poll_list_length(
            name, list_length, poll_frequency_ms, num_tries
        )

    @exception_handler
    def poll_list_length_gte(
        self, name: str, list_length: int, poll_frequency_ms: int, num_tries: int
    ) -> bool:
        """Poll list length until length is greater than or equal
        to the user-provided length. If maximum number of
        attempts is exceeded, false is returned.

        The aggregation list key used to check for list length
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_list_ensemble_prefix()
        for more details.

        :param name: The name of the list
        :type name: str
        :param list_length: The desired minimum length of the list
        :type list_length: int
        :param poll_frequency_ms: The time delay between checks, in milliseconds
        :type poll_frequency_ms: int
        :param num_tries: The total number of times to check for the name
        :type num_tries: int
        :return: Returns true if the list is found with a length greater
                 than or equal to the provided length, otherwise false
        :rtype: bool
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(name, "name", str)
        typecheck(list_length, "list_length", int)
        typecheck(poll_frequency_ms, "poll_frequency_ms", int)
        typecheck(num_tries, "num_tries", int)
        return self._client.poll_list_length_gte(
            name, list_length, poll_frequency_ms, num_tries
        )

    @exception_handler
    def poll_list_length_lte(
        self, name: str, list_length: int, poll_frequency_ms: int, num_tries: int
    ) -> bool:
        """Poll list length until length is less than or equal
        to the user-provided length. If maximum number of
        attempts is exceeded, false is returned.

        The aggregation list key used to check for list length
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_list_ensemble_prefix()
        for more details.

        :param name: The name of the list
        :type name: str
        :param list_length: The desired maximum length of the list
        :type list_length: int
        :param poll_frequency_ms: The time delay between checks, in milliseconds
        :type poll_frequency_ms: int
        :param num_tries: The total number of times to check for the name
        :type num_tries: int
        :return: Returns true if the list is found with a length less
                 than or equal to the provided length, otherwise false
        :rtype: bool
        :raises RedisReplyError: if there is an error
                in command execution.
        """
        typecheck(name, "name", str)
        typecheck(list_length, "list_length", int)
        typecheck(poll_frequency_ms, "poll_frequency_ms", int)
        typecheck(num_tries, "num_tries", int)
        return self._client.poll_list_length_lte(
            name, list_length, poll_frequency_ms, num_tries
        )

    @exception_handler
    def get_datasets_from_list(self, list_name: str) -> t.List[Dataset]:
        """Get datasets from an aggregation list

        The aggregation list key used to retrieve datasets
        may be formed by applying a prefix to the supplied
        name. See set_data_source() and use_list_ensemble_prefix()
        for more details.  An empty or nonexistant
        aggregation list returns an empty vector.

        :param list_name: The name of the list
        :type list_name: str
        :return: A list of DataSet objects.
        :rtype: list[DataSet]
        :raises RedisReplyError: if there is an error in command execution.
        """
        typecheck(list_name, "list_name", str)
        return self._client.get_datasets_from_list(list_name)

    @exception_handler
    def get_dataset_list_range(
        self, list_name: str, start_index: int, end_index: int
    ) -> t.List[Dataset]:
        """Get a range of datasets (by index) from an aggregation list

        The aggregation list key used to retrieve datasets
        may be formed by applying a prefix to the supplied
        name. See set_data_source()  and use_list_ensemble_prefix()
        for more details.  An empty or nonexistant aggregation
        list returns an empty vector.  If the provided
        end_index is beyond the end of the list, that index will
        be treated as the last index of the list.  If start_index
        and end_index are inconsistent (e.g. end_index is less
        than start_index), an empty list of datasets will be returned.

        :param list_name: The name of the list
        :type list_name: str
        :param start_index: The starting index of the range (inclusive,
               starting at zero).  Negative values are
               supported.  A negative value indicates offsets
               starting at the end of the list. For example, -1 is
               the last element of the list.
        :type start_index: int
        :param end_index: The ending index of the range (inclusive,
               starting at zero).  Negative values are
               supported.  A negative value indicates offsets
               starting at the end of the list. For example, -1 is
               the last element of the list.
        :return: A list of DataSet objects.
        :rtype: list[DataSet]
        :raises RedisReplyError: if there is an error in command execution.
        """
        typecheck(list_name, "list_name", str)
        typecheck(start_index, "start_index", int)
        typecheck(end_index, "end_index", int)
        return self._client.get_dataset_list_range(list_name, start_index, end_index)

    # ---- helpers --------------------------------------------------------

    @staticmethod
    def __check_tensor_args(
        inputs: t.Optional[t.Union[t.List[str], str]],
        outputs: t.Optional[t.Union[t.List[str], str]],
    ) -> t.Tuple[t.List[str], t.List[str]]:
        inputs = init_default([], inputs, (list, str))
        outputs = init_default([], outputs, (list, str))
        assert inputs is not None and outputs is not None
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        return inputs, outputs

    @staticmethod
    def __check_backend(backend: str) -> str:
        backend = backend.upper()
        if backend in ["TF", "TFLITE", "TORCH", "ONNX"]:
            return backend

        raise TypeError(f"Backend type {backend} unsupported")

    @staticmethod
    def __check_file(file: str) -> str:
        file_path = osp.abspath(file)
        if not osp.isfile(file_path):
            raise FileNotFoundError(file_path)
        return file_path

    @staticmethod
    def __check_device(device: str) -> str:
        device = device.upper()
        if not device.startswith("CPU") and not device.startswith("GPU"):
            raise TypeError("Device argument must start with either CPU or GPU")
        return device

    @staticmethod
    def __set_address(address: str) -> None:
        if "SSDB" in os.environ:
            del os.environ["SSDB"]
        os.environ["SSDB"] = address
