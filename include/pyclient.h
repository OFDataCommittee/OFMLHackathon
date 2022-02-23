/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2022, Hewlett Packard Enterprise
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

#define PY_CLIENT_H
#ifdef __cplusplus

#include "client.h"
#include "pydataset.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <unordered_map>

///@file

namespace SmartRedis {

namespace py = pybind11;

class PyClient;

/*!
*   \brief The PyClient class is a wrapper around the
           C++ client that is needed for the Python
           client.
*/
class PyClient
{
    public:

        /*!
        *   \brief PyClient constructor
        *   \param cluster Flag to indicate if a database cluster
        *                  is being used
        */
        PyClient(bool cluster);

        /*!
        *   \brief PyClient destructor
        */
        ~PyClient();

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
        *          specificed input and output tensors
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
                                const std::string& tag = "",
                                const std::vector<std::string>& inputs
                                    = std::vector<std::string>(),
                                const std::vector<std::string>& outputs
                                    = std::vector<std::string>());

        /*!
        *   \brief Run a model in the database using the
        *          specificed input and output tensors
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
        * \brief Set whether names of tensors or datasets should be
        *        prefixed (e.g. in an ensemble) to form database keys.
        *        Prefixes will only be used if they were previously set through
        *        the environment variables SSKEYOUT and SSKEYIN.
        *        Keys formed before this function is called will not be affected.
        *        By default, the client prefixes tensor and dataset keys
        *        with the first prefix specified with the SSKEYIN
        *        and SSKEYOUT environment variables.
        *
        * \param use_prefix If set to true, all future operations
        *                   on tensors and datasets will add
        *                   a prefix to the entity names, if available.
        */
        void use_tensor_ensemble_prefix(bool use_prefix);

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

    private:

        /*!
        *   \brief Pointer to a Client object for
        *          executing server commands
        */
        Client* _client;

};

} //namespace SmartRedis

#endif //PY_CLIENT_H
