# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

import inspect
import os
import os.path as osp
import functools

import numpy as np

from .dataset import Dataset
from .smartredisPy import PyClient
from .util import Dtypes, init_default, exception_handler, typecheck

from .error import *
from .smartredisPy import RedisReplyError as PybindRedisReplyError

class Client(PyClient):
    def __init__(self, address=None, cluster=False):
        """Initialize a RedisAI client

        For clusters, the address can be a single tcp/ip address and port
        of a database node. The rest of the cluster will be discovered
        by the client itself. (e.g. address="127.0.0.1:6379")

        If an address is not set, the client will look for the environment
        variable ``SSDB`` (e.g. SSDB="127.0.0.1:6379;")

        :param address: Address of the database
        :param cluster: True if connecting to a redis cluster, defaults to False
        :type cluster: bool, optional
        :raises RedisConnectionError: if connection initialization fails
        """
        if address:
            self.__set_address(address)
        if "SSDB" not in os.environ:
            raise RedisConnectionError("Could not connect to database. $SSDB not set")
        try:
            super().__init__(cluster)
        except PybindRedisReplyError as e:
            raise RedisConnectionError(str(e)) from None
        except RuntimeError as e:
            raise RedisConnectionError(str(e)) from None

    @exception_handler
    def put_tensor(self, name, data):
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
        super().put_tensor(name, dtype, data)

    @exception_handler
    def get_tensor(self, name):
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
        return super().get_tensor(name)

    @exception_handler
    def delete_tensor(self, name):
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
        super().delete_tensor(name)

    @exception_handler
    def copy_tensor(self, src_name, dest_name):
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
        super().copy_tensor(src_name, dest_name)

    @exception_handler
    def rename_tensor(self, old_name, new_name):
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
        super().rename_tensor(old_name, new_name)

    @exception_handler
    def put_dataset(self, dataset):
        """Put a Dataset instance into the database

        The final dataset key under which the dataset is stored
        is generated from the name that was supplied when the
        dataset was created and may be prefixed. See
        use_tensor_ensemble_prefix() for more details.

        All associated tensors and metadata within the Dataset
        instance will also be stored.

        :param dataset: a Dataset instance
        :type dataset: Dataset
        :raises TypeError: if argument is not a Dataset
        :raises RedisReplyError: if update fails
        """
        typecheck(dataset, "dataset", Dataset)
        pybind_dataset = dataset.get_data()
        super().put_dataset(pybind_dataset)

    @exception_handler
    def get_dataset(self, name):
        """Get a dataset from the database

        The dataset key used to locate the dataset
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param name: name the dataset is stored under
        :type name: str
        :raises RedisReplyError: if retrieval fails
        :return: Dataset instance
        :rtype: Dataset
        """
        typecheck(name, "name", str)
        dataset = super().get_dataset(name)
        python_dataset = Dataset.from_pybind(dataset)
        return python_dataset

    @exception_handler
    def delete_dataset(self, name):
        """Delete a dataset within the database

        The dataset key used to locate the dataset to be deleted
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param name: name of the dataset
        :type name: str
        :raises RedisReplyError: if deletion fails
        """
        typecheck(name, "name", str)
        super().delete_dataset(name)

    @exception_handler
    def copy_dataset(self, src_name, dest_name):
        """Copy a dataset from one key to another

        The source and destination dataset keys used to
        locate the dataset may be formed by applying prefixes
        to the supplied src_name and dest_name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param src_name: source name for dataset to be copied
        :type src_name: str
        :param dest_name: new name of dataset
        :type dest_name: str
        :raises RedisReplyError: if copy operation fails
        """
        typecheck(src_name, "src_name", str)
        typecheck(dest_name, "dest_name", str)
        super().copy_dataset(src_name, dest_name)

    @exception_handler
    def rename_dataset(self, old_name, new_name):
        """Rename a dataset in the database

        The old and new dataset keys used to find and relocate
        the dataset may be formed by applying prefixes to the supplied
        old_name and new_name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param old_name: original name of the dataset to be renamed
        :type old_name: str
        :param new_name: new name for the dataset
        :type new_name: str
        :raises RedisReplyError: if rename operation fails
        """
        typecheck(old_name, "old_name", str)
        typecheck(new_name, "new_name", str)
        super().rename_dataset(old_name, new_name)

    @exception_handler
    def set_function(self, name, function, device="CPU"):
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
            raise TypeError(f"Argument provided for function, {type(function)}, is not callable")
        device = self.__check_device(device)
        fn_src = inspect.getsource(function)
        super().set_script(name, device, fn_src)

    @exception_handler
    def set_script(self, name, script, device="CPU"):
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
        super().set_script(name, device, script)

    @exception_handler
    def set_script_from_file(self, name, file, device="CPU"):
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
        super().set_script_from_file(name, device, file_path)

    @exception_handler
    def get_script(self, name):
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
        script = super().get_script(name)
        return script

    @exception_handler
    def run_script(self, name, fn_name, inputs, outputs):
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
        :type inputs: list[str]
        :param outputs: database tensor names to receive script outputs
        :type outputs: list[str]
        :raises RedisReplyError: if script execution fails
        """
        typecheck(name, "name", str)
        typecheck(fn_name, "fn_name", str)
        typecheck(inputs, "inputs", list)
        typecheck(outputs, "outputs", list)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        super().run_script(name, fn_name, inputs, outputs)

    @exception_handler
    def get_model(self, name):
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
        model = super().get_model(name)
        return model

    @exception_handler
    def set_model(
        self,
        name,
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
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: list[str], optional
        :param outputs: model outputs (TF only), defaults to None
        :type outputs: list[str], optional
        :raises RedisReplyError: if model fails to set
        """
        typecheck(name, "name", str)
        typecheck(backend, "backend", str)
        typecheck(device, "device", str)
        typecheck(batch_size, "batch_size", int)
        typecheck(min_batch_size, "min_batch_size", int)
        typecheck(tag, "tag", str)
        device = self.__check_device(device)
        backend = self.__check_backend(backend)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        super().set_model(
            name,
            model,
            backend,
            device,
            batch_size,
            min_batch_size,
            tag,
            inputs,
            outputs,
        )

    @exception_handler
    def set_model_from_file(
        self,
        name,
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
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: list[str], optional
        :param outputs: model outupts (TF only), defaults to None
        :type outputs: list[str], optional
        :raises RedisReplyError: if model fails to set
        """
        typecheck(name, "name", str)
        typecheck(model_file, "model_file", str)
        typecheck(backend, "backend", str)
        typecheck(device, "device", str)
        typecheck(batch_size, "batch_size", int)
        typecheck(min_batch_size, "min_batch_size", int)
        typecheck(tag, "tag", str)
        device = self.__check_device(device)
        backend = self.__check_backend(backend)
        m_file = self.__check_file(model_file)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        super().set_model_from_file(
            name,
            m_file,
            backend,
            device,
            batch_size,
            min_batch_size,
            tag,
            inputs,
            outputs,
        )

    @exception_handler
    def run_model(self, name, inputs=None, outputs=None):
        """Execute a stored model

        The model key used to locate the model to be run
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_model_ensemble_prefix() for more details.

        :param name: name for stored model
        :type name: str
        :param inputs: names of stored inputs to provide model, defaults to None
        :type inputs: list[str], optional
        :param outputs: names to store outputs under, defaults to None
        :type outputs: list[str], optional
        :raises RedisReplyError: if model execution fails
        """
        typecheck(name, "name", str)
        inputs, outputs = self.__check_tensor_args(inputs, outputs)
        super().run_model(name, inputs, outputs)

    @exception_handler
    def tensor_exists(self, name):
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
        return super().tensor_exists(name)

    @exception_handler
    def dataset_exists(self, name):
        """Check if a dataset exists in the database

        The dataset key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

        :param name: The dataset name that will be checked in the database
        :type name: str
        :returns: Returns true if the dataset exists in the database
        :rtype: bool
        :raises RedisReplyError: if `dataset_exists` fails (i.e. causes an error)
        """
        typecheck(name, "name", str)
        return super().dataset_exists(name)

    @exception_handler
    def model_exists(self, name):
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
        return super().model_exists(name)

    @exception_handler
    def key_exists(self, key):
        """Check if the key exists in the database

        :param key: The key that will be checked in the database
        :type key: str
        :returns: Returns true if the key exists in the database
        :rtype: bool
        :raises RedisReplyError: if `key_exists` fails
        """
        typecheck(key, "key", str)
        return super().key_exists(key)

    @exception_handler
    def poll_key(self, key, poll_frequency_ms, num_tries):
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
        return super().poll_key(key, poll_frequency_ms, num_tries)

    @exception_handler
    def poll_tensor(self, name, poll_frequency_ms, num_tries):
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
        return super().poll_tensor(name, poll_frequency_ms, num_tries)

    @exception_handler
    def poll_dataset(self, name, poll_frequency_ms, num_tries):
        """Check if a dataset exists in the database

        The check is repeated at a specified polling interval and for
        a specified number of retries.
        The dataset key used to check for existence
        may be formed by applying a prefix to the supplied
        name. See set_data_source()
        and use_tensor_ensemble_prefix() for more details.

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
        return super().poll_dataset(name, poll_frequency_ms, num_tries)

    @exception_handler
    def poll_model(self, name, poll_frequency_ms, num_tries):
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
        return super().poll_model(name, poll_frequency_ms, num_tries)

    @exception_handler
    def set_data_source(self, source_id):
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
        return super().set_data_source(source_id)

    @exception_handler
    def use_model_ensemble_prefix(self, use_prefix):
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
        return super().use_model_ensemble_prefix(use_prefix)

    @exception_handler
    def use_tensor_ensemble_prefix(self, use_prefix):
        """Control whether tensor and dataset keys are
           prefixed (e.g. in an ensemble) when forming database keys

        This function can be used to avoid key collisions in an ensemble
        by prepending the string value from the environment variable SSKEYIN
        to tensor and dataset names.
        Prefixes will only be used if they were previously set through
        environment variables SSKEYIN and SSKEYOUT.
        Keys for entities created before this function is called
        will not be retroactively prefixed.
        By default, the client prefixes tensor and dataset
        keys when a prefix is available.

        :param use_prefix: If set to true, all future operations
                           on tensors and datasets will use a prefix, if
                           available.
        :type use_prefix: bool
        """
        typecheck(use_prefix, "use_prefix", bool)
        return super().use_tensor_ensemble_prefix(use_prefix)

    @exception_handler
    def get_db_node_info(self, addresses):
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
        return super().get_db_node_info(addresses)

    @exception_handler
    def get_db_cluster_info(self, addresses):
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
        return super().get_db_cluster_info(addresses)

    @exception_handler
    def get_ai_info(self, address, key, reset_stat=False):
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
        return super().get_ai_info(address, key, reset_stat)

    @exception_handler
    def flush_db(self, addresses):
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
        super().flush_db(addresses)

    @exception_handler
    def config_get(self, expression, address):
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
        return super().config_get(expression, address)

    @exception_handler
    def config_set(self, config_param, value, address):
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
        super().config_set(config_param, value, address)

    @exception_handler
    def save(self, addresses):
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
        super().save(addresses)

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
