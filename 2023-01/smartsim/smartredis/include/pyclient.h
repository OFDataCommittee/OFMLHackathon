/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Hewlett Packard Enterprise
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SMARTREDIS_PYCLIENT_H
#define SMARTREDIS_PYCLIENT_H

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <unordered_map>
#include "client.h"
#include "pydataset.h"
#include "pysrobject.h"
#include "pyconfigoptions.h"

///@file

namespace SmartRedis {

namespace py = pybind11;

/*!
*   \brief The PyClient class is a wrapper around the
           C++ client that is needed for the Python
           client.
*/
class PyClient : public PySRObject
{
    public:

        /*!
        *   \brief Simple constructor that uses default environment variables
        *          to locate configuration settings
        *   \param logger_name Identifier for the current client
        */
        PyClient(const std::string& logger_name);

        /*!
        *   \brief Constructor that uses a ConfigOptions object
        *          to locate configuration settings
        *   \param config_options The ConfigOptions object to use
        *   \param logger_name Identifier for the current client
        */
        PyClient(
            PyConfigOptions& config_options,
            const std::string& logger_name);

        /*!
        *   \brief PyClient constructor (deprecated)
        *   \param cluster Flag to indicate if a database cluster
        *                  is being used
        *   \param logger_name Identifier for the current client
        */
        PyClient(
            bool cluster,
            const std::string& logger_name = std::string("default"));

        /*!
        *   \brief PyClient destructor
        */
        virtual ~PyClient();

        /*!
        *   \brief Put a tensor into the database
        *   \param name The name to associate with this tensor
        *              in the database
        *   \param type The data type of the tensor
        *   \param data Numpy array with Pybind*
        *   \throw RuntimeException for all client errors
        */
        void put_tensor(std::string& name,
                        std::string& type,
                        py::array data);

        /*!
        *   \brief  Retrieve a tensor from the database.
        *   \details The memory of the data pointer used
        *            to construct the Numpy array is valid
        *            until the PyClient is destroyed.
        *            However, given that the memory
        *            associated with the return data is
        *            valid until PyClient destruction, this method
        *            should not be used repeatedly for large tensor
        *            data.  Instead it is recommended that the user
        *            use PyClient.unpack_tensor() for large tensor
        *            data and to limit memory use by the PyClient.
        *   \param name The name used to reference the tensor
        *   \throw RuntimeException for all client errors
        */
        py::array get_tensor(const std::string& name);

        /*!
        *   \brief delete a tensor stored in the database
        *   \param name The name of tensor to delete
        *   \throw RuntimeException for all client errors
        */
        void delete_tensor(const std::string& name);

        /*!
        *   \brief rename a tensor stored in the database
        *   \param old_name The original name of tensor to rename
        *   \param new_name the new name of the tensor
        *   \throw RuntimeException for all client errors
        */
        void rename_tensor(const std::string& old_name,
                           const std::string& new_name);

        /*!
        *   \brief copy a tensor to a new name
        *   \param src_name The name of tensor to copy
        *   \param dest_name the name to store tensor copy at
        *   \throw RuntimeException for all client errors
        */
        void copy_tensor(const std::string& src_name,
                         const std::string& dest_name);


        /*!
        *   \brief Send a PyDataSet object to the database
        *   \param dataset The PyDataSet object to send to the database
        *   \throw RuntimeException for all client errors
        */
        void put_dataset(PyDataset& dataset);


        /*!
        *   \brief Get a PyDataSet object from the database
        *   \param name The name of the dataset to retrieve
        *   \returns Pointer to the PyDataSet
        *            object retrieved from the database
        *   \throw RuntimeException for all client errors
        */
        PyDataset* get_dataset(const std::string& name);


        /*!
        *   \brief delete a dataset stored in the database
        *   \param name The name of dataset to delete
        *   \throw RuntimeException for all client errors
        */
        void delete_dataset(const std::string& name);

        /*!
        *   \brief rename a dataset stored in the database
        *   \param old_name The original name of dataset to rename
        *   \param new_name the new name for the dataset
        *   \throw RuntimeException for all client errors
        */
        void rename_dataset(const std::string& old_name,
                            const std::string& new_name);

        /*!
        *   \brief copy a dataset to a new name
        *   \param src_name The name of dataset to copy
        *   \param dest_name the name to store dataset copy at
        *   \throw RuntimeException for all client errors
        */
        void copy_dataset(const std::string& src_name,
                          const std::string& dest_name);


        /*!
        *   \brief Set a script from file in the
        *          database for future execution
        *   \param name The name to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script_file The source file for the script
        *   \throw RuntimeException for all client errors
        */
        void set_script_from_file(const std::string& name,
                                  const std::string& device,
                                  const std::string& script_file);

        /*!
        *   \brief Set a script from file in the database for future
        *          execution in a multi-GPU system
        *   \param name The name to associate with the script
        *   \param script_file The source file for the script
        *   \param first_gpu The first GPU (zero-based) to use for this script
        *   \param num_gpus The number of GPUs to use for this script
        *   \throw RuntimeException for all client errors
        */
        void set_script_from_file_multigpu(const std::string& name,
                                           const std::string& script_file,
                                           int first_gpu,
                                           int num_gpus);

        /*!
        *   \brief Set a script from std::string_view buffer in the
        *          database for future execution
        *   \param name The name to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script The script source in a std::string_view
        *   \throw RuntimeException for all client errors
        */
        void set_script(const std::string& name,
                        const std::string& device,
                        const std::string_view& script);

        /*!
        *   \brief Set a script from std::string_view buffer in the
        *          database for future execution in a multi-GPU system
        *   \param name The name to associate with the script
        *   \param script The script source in a std::string_view
        *   \param first_gpu The first GPU (zero-based) to use for this script
        *   \param num_gpus The number of GPUs to use for this script
        *   \throw RuntimeException for all client errors
        */
        void set_script_multigpu(const std::string& name,
                                 const std::string_view& script,
                                 int first_gpu,
                                 int num_gpus);

        /*!
        *   \brief Retrieve the script from the database
        *   \param name The name associated with the script
        *   \returns A std::string_view containing the script.
        *            The memory associated with the script
        *            is managed by the PyClient and is valid
        *            until the destruction of the PyClient.
        *   \throw RuntimeException for all client errors
        */
        std::string_view get_script(const std::string& name);

        /*!
        *   \brief Run a script function in the database using the
        *          specified input and output tensors
        *   \param name The name associated with the script
        *   \param function The name of the function in the script
        *                   to run
        *   \param inputs The names of input tensors to use
        *                 in the script
        *   \param outputs The names of output tensors that
        *                 will be used to save script results
        *   \throw RuntimeException for all client errors
        */
        void run_script(const std::string& name,
                        const std::string& function,
                        std::vector<std::string>& inputs,
                        std::vector<std::string>& outputs);

        /*!
        *   \brief Run a script function in the database using the
        *          specified input and output tensors in a multi-GPU system
        *   \param name The name associated with the script
        *   \param function The name of the function in the script to run
        *   \param inputs The names of input tensors to use in the script
        *   \param outputs The names of output tensors that will be used
        *                  to save script results
        *   \param offset index of the current image, such as a processor
        *                   ID or MPI rank
        *   \param first_gpu The first GPU (zero-based) to use for this script
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw RuntimeException for all client errors
        */
        void run_script_multigpu(const std::string& name,
                                 const std::string& function,
                                 std::vector<std::string>& inputs,
                                 std::vector<std::string>& outputs,
                                 int offset,
                                 int first_gpu,
                                 int num_gpus);

        /*!
        *   \brief Set a model from std::string_view buffer in the
        *          database for future execution
        *   \param name The name to associate with the model
        *   \param model The model as a continuous buffer string_view
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model
        *                         execution
        *   \param min_batch_timeout Max time (ms) to wait for min batch size
        *   \param tag A tag to attach to the model for
        *              information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        *   \throw RuntimeException for all client errors
        */
        void set_model(const std::string& name,
                        const std::string_view& model,
                        const std::string& backend,
                        const std::string& device,
                        int batch_size = 0,
                        int min_batch_size = 0,
                        int min_batch_timeout = 0,
                        const std::string& tag = "",
                        const std::vector<std::string>& inputs
                            = std::vector<std::string>(),
                        const std::vector<std::string>& outputs
                            = std::vector<std::string>());

        /*!
        *   \brief Set a model from std::string_view buffer in the
        *          database for future execution in a multi-GPU system
        *   \param name The name to associate with the model
        *   \param model The model as a continuous buffer string_view
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param first_gpu The first GPU (zero-based) to use for this model
        *   \param num_gpus The number of GPUs to use for this model
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model
        *                         execution
        *   \param min_batch_timeout Max time (ms) to wait for min batch size
        *   \param tag A tag to attach to the model for
        *              information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        *   \throw RuntimeException for all client errors
        */
        void set_model_multigpu(const std::string& name,
                                const std::string_view& model,
                                const std::string& backend,
                                int first_gpu,
                                int num_gpus,
                                int batch_size = 0,
                                int min_batch_size = 0,
                                int min_batch_timeout = 0,
                                const std::string& tag = "",
                                const std::vector<std::string>& inputs
                                    = std::vector<std::string>(),
                                const std::vector<std::string>& outputs
                                    = std::vector<std::string>());

        /*!
        *   \brief Set a model from file in the
        *          database for future execution
        *   \param name The name to associate with the model
        *   \param model_file The source file for the model
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model
        *                         execution
        *   \param min_batch_timeout Max time (ms) to wait for min batch size
        *   \param tag A tag to attach to the model for
        *              information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        *   \throw RuntimeException for all client errors
        */
        void set_model_from_file(const std::string& name,
                                const std::string& model_file,
                                const std::string& backend,
                                const std::string& device,
                                int batch_size = 0,
                                int min_batch_size = 0,
                                int min_batch_timeout = 0,
                                const std::string& tag = "",
                                const std::vector<std::string>& inputs
                                    = std::vector<std::string>(),
                                const std::vector<std::string>& outputs
                                    = std::vector<std::string>());

        /*!
        *   \brief Set a model from file in the database for future
        *          execution in a multi-GPU system
        *   \param name The name to associate with the model
        *   \param model_file The source file for the model
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param first_gpu The first GPU (zero-based) to use for this model
        *   \param num_gpus The number of GPUs to use for this model
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model
        *                         execution
        *   \param min_batch_timeout Max time (ms) to wait for min batch size
        *   \param tag A tag to attach to the model for
        *              information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        *   \throw RuntimeException for all client errors
        */
        void set_model_from_file_multigpu(const std::string& name,
                                const std::string& model_file,
                                const std::string& backend,
                                int first_gpu,
                                int num_gpus,
                                int batch_size = 0,
                                int min_batch_size = 0,
                                int min_batch_timeout = 0,
                                const std::string& tag = "",
                                const std::vector<std::string>& inputs
                                    = std::vector<std::string>(),
                                const std::vector<std::string>& outputs
                                    = std::vector<std::string>());

        /*!
        *   \brief Run a model in the database using the
        *          specified input and output tensors
        *   \param name The name associated with the model
        *   \param inputs The names of input tensors to use
        *                 in the model
        *   \param outputs The names of output tensors that
        *                 will be used to save model results
        *   \throw RuntimeException for all client errors
        */
        void run_model(const std::string& name,
                        std::vector<std::string> inputs,
                        std::vector<std::string> outputs);

        /*!
        *   \brief Run a model in the database using the
        *          specified input and output tensors in a multi-GPU system
        *   \param name The name associated with the model
        *   \param inputs The names of input tensors to use in the model
        *   \param outputs The names of output tensors that will be used
        *                  to save model results
        *   \param offset index of the current image, such as a processor
        *                   ID or MPI rank
        *   \param first_gpu The first GPU (zero-based) to use for this model
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw RuntimeException for all client errors
        */
        void run_model_multigpu(const std::string& name,
                                std::vector<std::string> inputs,
                                std::vector<std::string> outputs,
                                int offset,
                                int first_gpu,
                                int num_gpus);

        /*!
        *   \brief Remove a model from the database
        *   \param name The name associated with the model
        *   \throw RuntimeException for all client errors
        */
        void delete_model(const std::string& name);

        /*!
        *   \brief Remove a model from the database
        *   \param name The name associated with the model
        *   \param first_gpu the first GPU (zero-based) to use with the model
        *   \param num_gpus the number of gpus for which the model was stored
        *   \throw RuntimeException for all client errors
        */
        void delete_model_multigpu(const std::string& name, int first_gpu, int num_gpus);

        /*!
        *   \brief Remove a script from the database
        *   \param name The name associated with the script
        *   \throw RuntimeException for all client errors
        */
        void delete_script(const std::string& name);

        /*!
        *   \brief Remove a script from the database that was stored
        *          for use with multiple GPUs
        *   \param name The name associated with the script
        *   \param first_gpu the first GPU (zero-based) to use with the script
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw RuntimeException for all client errors
        */
        void delete_script_multigpu(const std::string& name, int first_gpu, int num_gpus);

        /*!
        *   \brief Retrieve the model from the database
        *   \param name The name associated with the model
        *   \returns A py:bytes object containing the model
        *   \throw RuntimeException for all client errors
        */
        py::bytes get_model(const std::string& name);

        /*!
        *   \brief Check if the key exists in the database
        *   \param key The key that will be checked in the database.
        *              No prefix will be added to \p key.
        *   \returns Returns true if the key exists in the database
        */
        bool key_exists(const std::string& key);

        /*!
        *   \brief Check if the tensor exists in the database
        *   \param name The name that will be checked in the database
        *               depending on the current prefixing
        *               behavior, the name will be automatically prefixed
        *               to form the corresponding key.
        *   \returns Returns true if the tensor exists in the database
        */
        bool tensor_exists(const std::string& name);

        /*!
        *   \brief Check if the dataset exists in the database
        *   \param name The name that will be checked in the database
        *               Depending on the current prefixing
        *               behavior, the name will be automatically prefixed
        *               to form the corresponding key.
        *   \returns Returns true if the dataset exists in the database
        */
        bool dataset_exists(const std::string& name);

        /*!
        *   \brief Check if the model or script exists in the database
        *   \param name The name that will be checked in the database
        *               depending on the current prefixing
        *               behavior, the name will be automatically prefixed
        *               to form the corresponding key.
        *   \returns Returns true if the model or script exists in the database
        */
        bool model_exists(const std::string& name);

        /*!
        *   \brief Check if the key exists in the database at a
        *          specified frequency for a specified number
        *          of times
        *   \param key The key that will be checked in the database
        *   \param poll_frequency_ms The frequency of checks for the
        *                            key in milliseconds
        *   \param num_tries The total number of times to check for the key.
        *   \returns Returns true if the key is found within the
        *            specified number of tries, otherwise false.
        */
        bool poll_key(const std::string& key,
                      int poll_frequency_ms,
                      int num_tries);

        /*!
        *   \brief Check if a tensor exists in the database at a
        *          specified frequency for a specified number
        *          of times. The name will be automatically prefixed
        *          base on prefixing behavior.
        *   \param name The name that will be checked in the database
        *   \param poll_frequency_ms The frequency of checks for the
        *                            name in milliseconds
        *   \param num_tries The total number of times to check for
        *                    the tensor.
        *   \returns Returns true if the tensor is found within the
        *            specified number of tries, otherwise false.
        */
        bool poll_tensor(const std::string& name,
                         int poll_frequency_ms,
                         int num_tries);

        /*!
        *   \brief Check if a dataset exists in the database at a
        *          specified frequency for a specified number
        *          of times. The name will be automatically prefixed
        *          base on prefixing behavior.
        *   \param name The name that will be checked in the database
        *               Depending on the current prefixing behavior,
        *               the name could be automatically prefixed
        *               to form the corresponding name.
        *   \param poll_frequency_ms The frequency of checks for the
        *                            name in milliseconds
        *   \param num_tries The total number of times to check for
        *                    the dataset.
        *   \returns Returns true if the dataset is found within the
        *            specified number of tries, otherwise false.
        */
        bool poll_dataset(const std::string& name,
                          int poll_frequency_ms,
                          int num_tries);

        /*!
        *   \brief Check if a model or script exists in the database at a
        *          specified frequency for a specified number
        *          of times. The name will be automatically prefixed
        *          base on prefixing behavior.
        *   \param name The name that will be checked in the database
        *   \param poll_frequency_ms The frequency of checks for the
        *                            name in milliseconds
        *   \param num_tries The total number of times to check for
        *                    the model or script.
        *   \returns Returns true if the model or script is found within the
        *            specified number of tries, otherwise false.
        */
        bool poll_model(const std::string& name,
                        int poll_frequency_ms,
                        int num_tries);

        /*!
        *   \brief Set the data source (i.e. key prefix for
        *          get functions)
        *   \param source_id The prefix for retrieval commands
        */
        void set_data_source(const std::string& source_id);

        /*!
        * \brief Set whether names of model or scripts should be
        *        prefixed (e.g. in an ensemble) to form database keys.
        *        Prefixes will only be used if they were previously set through
        *        the environment variables SSKEYOUT and SSKEYIN.
        *        Keys formed before this function is called will not be affected.
        *        By default, the client does not prefix model and script keys.
        *
        * \param use_prefix If set to true, all future operations
        *                   on model and scripts will add
        *                   a prefix to the entity names, if available.
        */
        void use_model_ensemble_prefix(bool use_prefix);

        /*!
        *   \brief Control whether aggregation lists are prefixed
        *   \details This function can be used to avoid key collisions in an
        *            ensemble by prepending the string value from the
        *            environment variable SSKEYIN and/or SSKEYOUT to
        *            aggregation list names.  Prefixes will only be used if
        *            they were previously set through the environment variables
        *            SSKEYOUT and SSKEYIN. Keys for aggregation lists created
        *            before this function is called will not be retroactively
        *            prefixed. By default, the client prefixes aggregation
        *            list keys with the first prefix specified with the SSKEYIN
        *            and SSKEYOUT environment variables.  Note that
        *            use_dataset_ensemble_prefix() controls prefixing
        *            for the entities in the aggregation list, and
        *            use_dataset_ensemble_prefix() should be given the
        *            same value that was used during the initial
        *            setting of the DataSet into the database.
        *  \param use_prefix If set to true, all future operations
        *                    on aggregation lists will use
        *                    a prefix, if available.
        */
        void use_list_ensemble_prefix(bool use_prefix);

        /*!
        * \brief Set whether names of tensors should be prefixed (e.g.
        *        in an ensemble) to form database keys.
        *        Prefixes will only be used if they were previously set through
        *        the environment variables SSKEYOUT and SSKEYIN.
        *        Keys formed before this function is called will not be affected.
        *        By default, the client prefixes tensor keys with the first
        *        prefix specified with the SSKEYIN and SSKEYOUT environment
        *        variables.
        *
        * \param use_prefix If set to true, all future operations on tensors will
        *                   add a prefix to the entity names, if available.
        */
        void use_tensor_ensemble_prefix(bool use_prefix);

        /*!
        * \brief Set whether names of datasets should be prefixed (e.g.
        *        in an ensemble) to form database keys.
        *        Prefixes will only be used if they were previously set through
        *        the environment variables SSKEYOUT and SSKEYIN.
        *        Keys formed before this function is called will not be affected.
        *        By default, the client prefixes tensor keys with the first
        *        prefix specified with the SSKEYIN and SSKEYOUT environment
        *        variables.
        *
        * \param use_prefix If set to true, all future operations on datasets will
        *                   add a prefix to the entity names, if available.
        */
        void use_dataset_ensemble_prefix(bool use_prefix);

        /*!
        *   \brief Returns information about the given database nodes
        *   \param addresses The addresses of the database nodes. Each address is
        *                    formatted as address:port e.g. 127.0.0.1:6379
        *   \returns A list of parsed_map objects containing all the
        *            information about the given database nodes
        *   \throw RuntimeException if the address is not addressable by this
        *          client.  In the case of using a cluster of database nodes,
        *          it is best practice to bind each node in the cluster
        *          to a specific adddress to avoid inconsistencies in
        *          addresses retreived with the CLUSTER SLOTS command.
        *          Inconsistencies in node addresses across
        *          CLUSTER SLOTS comands will lead to SRRuntimeException
        *          being thrown.
        */
        std::vector<py::dict> get_db_node_info(std::vector<std::string> addresses);

        /*!
        *   \brief \brief Returns the CLUSTER INFO command reply addressed to one
        *                 or multiple cluster nodes.
        *   \param addresses The addresses of the database nodes. Each address is
        *                    formatted as address:port e.g. 127.0.0.1:6379
        *   \returns A list of parsed_map objects containing all the cluster
        *            information about the given database nodes
        *   \throw RuntimeException if the address is not addressable by this
        *          client.  In the case of using a cluster of database nodes,
        *          it is best practice to bind each node in the cluster
        *          to a specific adddress to avoid inconsistencies in
        *          addresses retreived with the CLUSTER SLOTS command.
        *          Inconsistencies in node addresses across
        *          CLUSTER SLOTS comands will lead to SRRuntimeException
        *          being thrown.
        */
        std::vector<py::dict> get_db_cluster_info(std::vector<std::string> addresses);

        /*!
        *   \brief Returns the AI.INFO command reply from the database
        *          shard at the provided address
        *   \param addresses std::vector of addresses of the database
        *                    nodes (host:port)
        *   \param key The model or script key
        *   \param reset_stat Boolean indicating if the counters associated
        *                     with the model or script should be reset.
        *   \returns A std::vector of dictionaries that map AI.INFO
        *            field names to values for each of the provided
        *            database addresses.
        *   \throw SmartRedis::RuntimeException or derivative error object
        *          if command execution or reply parsing fails.
        */
        std::vector<py::dict>
        get_ai_info(const std::vector<std::string>& addresses,
                    const std::string& key,
                    const bool reset_stat);

        /*!
        *   \brief Delete all the keys of the given database
        *   \param addresses The addresses of the database nodes. Each address is
        *                    formatted as address:port e.g. 127.0.0.1:6379
        */
        void flush_db(std::vector<std::string> addresses);

        /*!
        *   \brief Read the configuration parameters of a running server.
        *   \param  expression Parameter used in the configuration or a
        *                      glob pattern (Use '*' to retrieve all
        *                      configuration parameters)
        *   \param address The address of the database node execute on
        *   \returns A dictionary that maps configuration parameters to their values
        *            If the provided expression does not exist, then an empty
        *            dictionary is returned.
        *   \throw RuntimeException if the address is not addressable by this
        *          client
        */
        py::dict config_get(std::string expression,std::string address);

        /*!
        *   \brief Reconfigure the server. It can change both trivial
        *          parameters or switch from one to another persistence option.
        *          All the configuration parameters set using this command are
        *          immediately loaded by Redis and will take effect starting with
        *          the next command executed.
        *   \param config_param A configuration parameter to set
        *   \param value The value to assign to the configuration parameter
        *   \param address The address of the database node execute on
        *   \throw RuntimeException if the address is not addressable by this
        *          client or if command fails to execute or if the config_param
        *          is unsupported.
        */
        void config_set(std::string config_param, std::string value, std::string address);

        /*!
        *   \brief Performs a synchronous save of the database shard producing a point in
        *          time snapshot of all the data inside the Redis instance  in the form of
        *          an RDB file.
        *   \param addresses The addressees of database nodes (host:port)
        *   \throw RuntimeException if the address is not addressable by this
        *          client or if command fails to execute
        */
        void save(std::vector<std::string> addresses);

        /*!
        *   \brief Appends a dataset to the aggregation list
        *   \details When appending a dataset to an aggregation list,
        *            the list will automatically be created if it does not
        *            exist (i.e. this is the first entry in the list).
        *            Aggregation lists work by referencing the dataset
        *            by storing its key, so appending a dataset
        *            to an aggregation list does not create a copy of the
        *            dataset.  Also, for this reason, the dataset
        *            must have been previously placed into the database
        *            with a separate call to put_dataset().
        *   \param list_name The name of the aggregation list
        *   \param dataset The DataSet to append
        *   \throw SmartRedis::Exception if the command fails
        */
        void append_to_list(const std::string& list_name,
                            PyDataset& dataset);

        /*!
        *   \brief Delete an aggregation list
        *   \details The key used to locate the aggregation list to be
        *            deleted may be formed by applying a prefix to the
        *            supplied name. See set_data_source()
        *            and use_list_ensemble_prefix() for more details.
        *   \param list_name The name of the aggregation list
        *   \throw SmartRedis::Exception if the command fails
        */
        void delete_list(const std::string& list_name);

        /*!
        *   \brief Copy an aggregation list
        *   \details The source and destination aggregation list keys used to
        *            locate and store the aggregation list may be formed by
        *            applying prefixes to the supplied src_name and dest_name.
        *            See set_data_source() and use_list_ensemble_prefix()
        *            for more details.
        *   \param src_name The source list name
        *   \param dest_name The destination list name
        *   \throw SmartRedis::Exception if the command fails
        */
        void copy_list(const std::string& src_name,
                       const std::string& dest_name);

        /*!
        *   \brief Rename an aggregation list
        *   \details The initial and target aggregation list key used to find and
        *            relocate the list may be formed by applying prefixes to
        *            the supplied src_name and dest_name. See set_data_source()
        *            and use_list_ensemble_prefix() for more details.
        *   \param src_name The initial list name
        *   \param dest_name The target list name
        *   \throw SmartRedis::Exception if the command fails
        */
        void rename_list(const std::string& src_name,
                         const std::string& dest_name);

        /*!
        *   \brief Get the number of entries in the list
        *   \param list_name The list name
        *   \throw SmartRedis::Exception if the command fails
        */
        int get_list_length(const std::string& list_name);

        /*!
        *   \brief Poll list length until length is equal
        *          to the provided length.  If maximum number of
        *          attempts is exceeded, false is returned.
        *   \details The aggregation list key used to check for list length
        *            may be formed by applying a prefix to the supplied
        *            name. See set_data_source() and use_list_ensemble_prefix()
        *            for more details.
        *   \param name The name of the list
        *   \param list_length The desired length of the list
        *   \param poll_frequency_ms The time delay between checks,
        *                            in milliseconds
        *   \param num_tries The total number of times to check for the name
        *   \returns Returns true if the list is found with a length greater
        *            than or equal to the provided length, otherwise false
        *   \throw SmartRedis::Exception if poll list length command fails
        */
        bool poll_list_length(const std::string& name, int list_length,
                              int poll_frequency_ms, int num_tries);

        /*!
        *   \brief Poll list length until length is greater than or equal
        *          to the user-provided length. If maximum number of
        *          attempts is exceeded, false is returned.
        *   \details The aggregation list key used to check for list length
        *            may be formed by applying a prefix to the supplied
        *            name. See set_data_source() and use_list_ensemble_prefix()
        *            for more details.
        *   \param name The name of the list
        *   \param list_length The desired minimum length of the list
        *   \param poll_frequency_ms The time delay between checks,
        *                            in milliseconds
        *   \param num_tries The total number of times to check for the name
        *   \returns Returns true if the list is found with a length greater
        *            than or equal to the provided length, otherwise false
        *   \throw SmartRedis::Exception if poll list length command fails
        */
        bool poll_list_length_gte(const std::string& name, int list_length,
                                  int poll_frequency_ms, int num_tries);

        /*!
        *   \brief Poll list length until length is less than or equal
        *          to the user-provided length. If maximum number of
        *          attempts is exceeded, false is returned.
        *   \details The aggregation list key used to check for list length
        *            may be formed by applying a prefix to the supplied
        *            name. See set_data_source() and use_list_ensemble_prefix()
        *            for more details.
        *   \param name The name of the list
        *   \param list_length The desired maximum length of the list
        *   \param poll_frequency_ms The time delay between checks,
        *                            in milliseconds
        *   \param num_tries The total number of times to check for the name
        *   \returns Returns true if the list is found with a length less
        *            than or equal to the provided length, otherwise false
        *   \throw SmartRedis::Exception if poll list length command fails
        */
        bool poll_list_length_lte(const std::string& name, int list_length,
                                  int poll_frequency_ms, int num_tries);

        /*!
        *   \brief Get datasets from an aggregation list
        *   \details The aggregation list key used to retrieve datasets
        *            may be formed by applying a prefix to the supplied
        *            name. See set_data_source() and use_list_ensemble_prefix()
        *            for more details.  An empty or nonexistant
        *            aggregation list returns an empty vector.
        *   \param list_name The name of the aggregation list
        *   \returns A vector containing DataSet objects.
        *   \throw SmartRedis::Exception if retrieval fails.
        */
        py::list get_datasets_from_list(const std::string& list_name);

        /*!
        *   \brief Get a range of datasets (by index) from an aggregation list
        *   \details The aggregation list key used to retrieve datasets
        *            may be formed by applying a prefix to the supplied
        *            name. See set_data_source()  and use_list_ensemble_prefix()
        *            for more details.  An empty or nonexistant aggregation
        *            list returns an empty vector.  If the provided
        *            end_index is beyond the end of the list, that index will
        *            be treated as the last index of the list.  If start_index
        *            and end_index are inconsistent (e.g. end_index is less
        *            than start_index), an empty list of datasets will be returned.
        *   \param list_name The name of the aggregation list
        *   \param start_index The starting index of the range (inclusive,
        *                      starting at zero).  Negative values are
        *                      supported.  A negative value indicates offsets
        *                      starting at the end of the list. For example, -1 is
        *                      the last element of the list.
        *   \param end_index The ending index of the range (inclusive,
        *                    starting at zero).  Negative values are
        *                    supported.  A negative value indicates offsets
        *                    starting at the end of the list. For example, -1 is
        *                    the last element of the list.
        *   \returns A vector containing DataSet objects.
        *   \throw SmartRedis::Exception if retrieval fails or
        *          input parameters are invalid
        */
        py::list get_dataset_list_range(const std::string& list_name,
                                        const int start_index,
                                        const int end_index);

        /*!
        *   \brief Reconfigure the chunking size that Redis uses for model
        *          serialization, replication, and the model_get command.
        *   \details This method triggers the AI.CONFIG method in the Redis
        *            database to change the model chunking size.
        *
        *            NOTE: The default size of 511MB should be fine for most
        *            applications, so it is expected to be very rare that a
        *            client calls this method. It is not necessary to call
        *            this method a model to be chunked.
        *   \param chunk_size The new chunk size in bytes
        *   \throw SmartRedis::Exception if the command fails.
        */
        void set_model_chunk_size(int chunk_size);

        /*!
        *   \brief Create a string representation of the Client
        *   \returns A string representation of the Client
        */
        std::string to_string();


    private:

        /*!
        *   \brief Pointer to a Client object for
        *          executing server commands
        */
        Client* _client;

};

} // namespace SmartRedis

#endif // SMARTREDIS_PYCLIENT_H
