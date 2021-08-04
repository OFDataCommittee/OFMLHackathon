/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Hewlett Packard Enterprise
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

#ifndef SMARTREDIS_CPP_CLIENT_H
#define SMARTREDIS_CPP_CLIENT_H
#ifdef __cplusplus
#include "string.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include "redisserver.h"
#include "rediscluster.h"
#include "redis.h"
#include "dataset.h"
#include "sharedmemorylist.h"
#include "command.h"
#include "commandlist.h"
#include "commandreply.h"
#include "commandreplyparser.h"
#include "tensorbase.h"
#include "tensor.h"
#include "enums/cpp_tensor_type.h"
#include "enums/cpp_memory_layout.h"

///@file

namespace SmartRedis {

class Client;

typedef redisReply ReplyElem;

///@file
/*!
*   \brief The Client class is the primary user-facing
*          class for executing server commands.
*/
class Client
{

    public:

        /*!
        *   \brief Client constructor
        *   \param cluster Flag to indicate if a database cluster
        *                  is being used
        */
        Client(bool cluster);

        /*!
        *   \brief Client copy constructor is not available
        */
        Client(const Client& client) = delete;

        /*!
        *   \brief Client move constructor
        */
        Client(Client&& client) = default;

        /*!
        *   \brief Client copy assignment operator
        *          is not available.
        */
        Client& operator=(const Client& client) = delete;

        /*!
        *   \brief Client move assignment operator
        */
        Client& operator=(Client&& client) = default;

        /*!
        *   \brief Client destructor
        */
        ~Client();

        /*!
        *   \brief Send a DataSet object to the database
        *   \param dataset The DataSet object to send to the database
        */
        void put_dataset(DataSet& dataset);

        /*!
        *   \brief Get a DataSet object from the database
        *   \param name The name of the dataset to retrieve
        *   \returns DataSet object retrieved from the database
        */
        DataSet get_dataset(const std::string& name);

        /*!
        *   \brief Move a DataSet to a new key.  A tensors
        *          and metdata in the DataSet will be moved.
        *   \param name The original key associated with the
        *               DataSet in the database
        *   \param new_name The new key to assign to the
        *                   DataSet object
        */
        void rename_dataset(const std::string& name,
                            const std::string& new_name);

        /*!
        *   \brief Copy a DataSet to a new key in the database.
        *          All tensors and metdata in the DataSet will
        *          be copied.
        *   \param src_name The key associated with the DataSet
        *                   that is to be copied
        *   \param dest_name The key in the database that will
        *                    store the copied database.
        */
        void copy_dataset(const std::string& src_name,
                          const std::string& dest_name);


        /*!
        *   \brief Delete a DataSet from the database.  All
        *          tensors and metdata in the DataSet will be
        *          deleted.
        *   \param name The name of the DataSet that should be
        *               deleted.
        */
        void delete_dataset(const std::string& name);


        /*!
        *   \brief Put a tensor into the database
        *   \param key The key to associate with this tensor
        *              in the database
        *   \param data A c-ptr to the beginning of the tensor data
        *   \param dims The dimensions of the tensor
        *   \param type The data type of the tensor
        *   \param mem_layout The memory layout of the provided
        *                     tensor data
        */
        void put_tensor(const std::string& key,
                        void* data,
                        const std::vector<size_t>& dims,
                        const TensorType type,
                        const MemoryLayout mem_layout);

        /*!
        *   \brief Get the tensor data, dimensions,
        *          and type for the provided tensor key.
        *          This function will allocate and retain
        *          management of the memory for the tensor
        *          data.
        *   \details The memory of the data pointer is valid
        *            until the Client is destroyed. This method
        *            is meant to be used when the dimensions and
        *            type of the tensor are unknown or the user does
        *            not want to manage memory.  However, given that
        *            the memory associated with the return data is
        *            valid until Client destruction, this method
        *            should not be used repeatedly for large tensor
        *            data.  Instead  it is recommended that the user
        *            use unpack_tensor() for large tensor data and
        *            to limit memory use by the Client.
        *   \param key  The name used to reference the tensor
        *   \param data A c-ptr reference that will be pointed to
        *               newly allocated memory
        *   \param dims A reference to a dimensions vector
        *               that will be filled with the retrieved
        *               tensor dimensions
        *   \param type A reference to a TensorType enum that will
        *               be set to the value of the retrieved
        *               tensor type
        *   \param mem_layout The MemoryLayout that the newly
        *                     allocated memory should conform to
        */
        void get_tensor(const std::string& key,
                        void*& data,
                        std::vector<size_t>& dims,
                        TensorType& type,
                        const MemoryLayout mem_layout);


        /*!
        *   \brief Get the tensor data, dimensions,
        *          and type for the provided tensor key.
        *          This function will allocate and retain
        *          management of the memory for the tensor
        *          data and dimensions.  This is a c-style
        *          interface for the tensor dimensions.  Another
        *          function exists for std::vector dimensions.
        *   \details The memory of the data pointer is valid
        *            until the Client is destroyed. This method
        *            is meant to be used when the dimensions and
        *            type of the tensor are unknown or the user does
        *            not want to manage memory.  However, given that the
        *            memory associated with the return data is valid
        *            until Client destruction, this method should not
        *            be used repeatedly for large tensor data.  Instead
        *            it is recommended that the user use unpack_tensor()
        *            for large tensor data and to limit memory use by
        *            the Client.
        *   \param key  The name used to reference the tensor
        *   \param data A c-ptr reference that will be pointed to
        *               newly allocated memory
        *   \param dims A reference to a c-ptr that will be
        *               pointed newly allocated memory
        *               that will be filled with the tensor dimensions
        *   \param n_dims A reference to type size_t variable
        *                 that will be set to the number
        *                 of tensor dimensions
        *   \param type A reference to a TensorType enum that will
        *               be set to the value of the retrieved
        *               tensor type
        *   \param mem_layout The MemoryLayout that the newly
        *                     allocated memory should conform to
        */
        void get_tensor(const std::string& key,
                        void*& data,
                        size_t*& dims,
                        size_t& n_dims,
                        TensorType& type,
                        const MemoryLayout mem_layout);

        /*!
        *   \brief Get tensor data and fill an already allocated
        *          array memory space that has the specified
        *          MemoryLayout.  The provided type and dimensions
        *          are checked against retrieved values to ensure
        *          the provided memory space is sufficient.  This
        *          method is the most memory efficient way
        *          to retrieve tensor data.
        *   \param key  The name used to reference the tensor
        *   \param data A c-ptr to the memory space to be filled
        *               with tensor data
        *   \param dims The dimensions of the memory space
        *   \param type The TensorType matching the data
        *               type of the memory space
        *   \param mem_layout The MemoryLayout of the provided
        *               memory space.
        */
        void unpack_tensor(const std::string& key,
                           void* data,
                           const std::vector<size_t>& dims,
                           const TensorType type,
                           const MemoryLayout mem_layout);

        /*!
        *   \brief Move a tensor from one key to another key
        *   \param key The original tensor key
        *   \param new_key The new tensor key
        */
        void rename_tensor(const std::string& key,
                           const std::string& new_key);

        /*!
        *   \brief Delete a tensor from the database
        *   \param key The key of tensor to delete
        */
        void delete_tensor(const std::string& key);

        /*!
        *   \brief Copy the tensor from the source
        *          key to the destination key
        *   \param src_key The key of the tensor to copy
        *   \param dest_key The destination key of the tensor
        */
        void copy_tensor(const std::string& src_key,
                         const std::string& dest_key);

        /*!
        *   \brief Set a model from file in the
        *          database for future execution
        *   \param key The key to associate with the model
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
        */
        void set_model_from_file(const std::string& key,
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
        *   \brief Set a model from std::string_view buffer in the
        *          database for future execution
        *   \param key The key to associate with the model
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
        */
        void set_model(const std::string& key,
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
        *   \brief Retrieve the model from the database
        *   \param key The key associated with the model
        *   \returns A std::string_view containing the model.
        *            The memory associated with the model
        *            is managed by the Client and is valid
        *            until the destruction of the Client.
        */
        std::string_view get_model(const std::string& key);

        /*!
        *   \brief Set a script from file in the
        *          database for future execution
        *   \param key The key to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script_file The source file for the script
        */
        void set_script_from_file(const std::string& key,
                                  const std::string& device,
                                  const std::string& script_file);

        /*!
        *   \brief Set a script from std::string_view buffer in the
        *          database for future execution
        *   \param key The key to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script The script source in a std::string_view
        */
        void set_script(const std::string& key,
                        const std::string& device,
                        const std::string_view& script);

        /*!
        *   \brief Retrieve the script from the database
        *   \param key The key associated with the script
        *   \returns A std::string_view containing the script.
        *            The memory associated with the script
        *            is managed by the Client and is valid
        *            until the destruction of the Client.
        */
        std::string_view get_script(const std::string& key);

        /*!
        *   \brief Run a model in the database using the
        *          specificed input and output tensors
        *   \param key The key associated with the model
        *   \param inputs The keys of inputs tensors to use
        *                 in the model
        *   \param outputs The keys of output tensors that
        *                 will be used to save model results
        */
        void run_model(const std::string& key,
                       std::vector<std::string> inputs,
                       std::vector<std::string> outputs);

        /*!
        *   \brief Run a script function in the database using the
        *          specificed input and output tensors
        *   \param key The key associated with the script
        *   \param function The name of the function in the script to run
        *   \param inputs The keys of inputs tensors to use
        *                 in the script
        *   \param outputs The keys of output tensors that
        *                 will be used to save script results
        */
        void run_script(const std::string& key,
                        const std::string& function,
                        std::vector<std::string> inputs,
                        std::vector<std::string> outputs);

        /*!
        *   \brief Check if the key exists in the database
        *   \param key The key that will be checked in the database.
        *              No prefix will be added to \p key.
        *   \returns Returns true if the key exists in the database
        */
        bool key_exists(const std::string& key);

        /*!
        *   \brief Check if the model (or the script) exists in the database
        *   \param name The name that will be checked in the database
        *               depending on the current prefixing behavior,
        *               the name could be automatically prefixed
        *               to form the corresponding key.
        *   \returns Returns true if the key exists in the database
        */
        bool model_exists(const std::string& name);

        /*!
        *   \brief Check if the tensor (or the dataset) exists in the database
        *   \param name The name that will be checked in the database
        *               depending on the current prefixing behavior,
        *               the name could be automatically prefixed
        *               to form the corresponding key.
        *   \returns Returns true if the key exists in the database
        */
        bool tensor_exists(const std::string& name);

        /*!
        *   \brief Check if the key exists in the database at a
        *          specified frequency for a specified number
        *          of times
        *   \param key The key that will be checked in the database
        *   \param poll_frequency_ms The frequency of checks for the
        *                            key in milliseconds
        *   \param num_tries The total number of times to check for
        *                    the specified number of keys.  If the
        *                    value is set to -1, the key will be
        *                    polled indefinitely.
        *   \returns Returns true if the key is found within the
        *            specified number of tries, otherwise false.
        */
        bool poll_key(const std::string& key,
                      int poll_frequency_ms,
                      int num_tries);

        /*!
        *   \brief Check if the tensor (or dataset) exists in the database
        *          at a specified frequency for a specified number
        *          of times
        *   \param name The key that will be checked in the database
        *               Depending on the current prefixing behavior,
        *               the name could be automatically prefixed
        *               to form the corresponding key.
        *   \param poll_frequency_ms The frequency of checks for the
        *                            key in milliseconds
        *   \param num_tries The total number of times to check for
        *                    the specified number of keys.  If the
        *                    value is set to -1, the key will be
        *                    polled indefinitely.
        *   \returns Returns true if the key is found within the
        *            specified number of tries, otherwise false.
        */
        bool poll_tensor(const std::string& name,
                         int poll_frequency_ms,
                         int num_tries);

        /*!
        *   \brief Check if the model (or script) exists in the database
        *          at a specified frequency for a specified number
        *          of times.
        *   \param name The name that will be checked in the database
        *               depending on the current prefixing behavior,
        *               the name could be automatically prefixed
        *               to form the corresponding key.
        *   \param poll_frequency_ms The frequency of checks for the
        *                            key in milliseconds
        *   \param num_tries The total number of times to check for
        *                    the specified number of keys.  If the
        *                    value is set to -1, the key will be
        *                    polled indefinitely.
        *   \returns Returns true if the key is found within the
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
        void set_data_source(std::string source_id);

        /*!
        * \brief Set whether names of tensor and dataset entities should be
        *        prefixed (e.g. in an ensemble) to form database keys.
        *        Prefixes will only be used if they were previously set through
        *        the environment variables SSKEYOUT and SSKEYIN.
        *        Keys of entities created before this function is called
        *        will not be affected.
        *        By default, the client prefixes tensor and dataset keys
        *        with the first prefix specified with the SSKEYIN
        *        and SSKEYOUT environment variables.
        *
        * \param use_prefix If set to true, all future operations
        *                   on tensors and datasets will use
        *                   a prefix, if available.
        */
        void use_tensor_ensemble_prefix(bool use_prefix);

        /*!
        * \brief Set whether names of model and script entities should be
        *        prefixed (e.g. in an ensemble) to form database keys.
        *        Prefixes will only be used if they were previously set through
        *        the environment variables SSKEYOUT and SSKEYIN.
        *        Keys of entities created before this function is called
        *        will not be affected.
        *        By default, the client does not prefix model and script keys.
        *
        * \param use_prefix If set to true, all future operations
        *                   on models and scripts will use
        *                   a prefix, if available.
        */
        void use_model_ensemble_prefix(bool use_prefix);

        /*!
        *   \brief Returns information about the given database node
        *   \param address The address of the database node (host:port)
        *   \returns parsed_reply_nested_map containing the database node information
	*   \throws std::runtime_error if the address is not addressable by this
	*           client.  In the case of using a cluster of database nodes,
	*           it is best practice to bind each node in the cluster
	*           to a specific adddress to avoid inconsistencies in
	*           addresses retreived with the CLUSTER SLOTS command.
	*           Inconsistencies in node addresses across
	*           CLUSTER SLOTS comands will lead to std::runtime_error
	*           being thrown.
        */
        parsed_reply_nested_map get_db_node_info(std::string address);

        /*!
        *   \brief Returns the CLUSTER INFO command reply addressed to a single
        *          cluster node.
        *   \param address The address of the database node (host:port)
        *   \returns parsed_reply_map containing the database cluster information.
        *            If this command is executed on a non-cluster database, an
        *            empty parsed_reply_map is returned.
	*   \throws std::runtime_error if the address is not addressable by this
        *           client.  In the case of using a cluster of database nodes,
        *           it is best practice to bind each node in the cluster
        *           to a specific adddress to avoid inconsistencies in
        *           addresses retreived with the CLUSTER SLOTS command.
        *           Inconsistencies in node addresses across
        *           CLUSTER SLOTS comands will lead to std::runtime_error
        *           being thrown.
        */
        parsed_reply_map get_db_cluster_info(std::string address);

    protected:

        /*!
        *  \brief Abstract base class used to generalized
        *         running with cluster or non-cluster
        */
        RedisServer* _redis_server;

        /*!
        *  \brief A pointer to a dynamically allocated
        *         RedisCluster object if the Client is
        *         being run in cluster mode.  This
        *         object will be destroyed with the Client.
        */
        RedisCluster* _redis_cluster;

        /*!
        *  \brief A pointer to a dynamically allocated
        *         Redis object if the Client is
        *         being run in cluster mode.  This
        *         object will be destroyed with the Client.
        */
        Redis* _redis;

        /*!
        *   \brief Execute a Command
        *   \param cmd The Command to execute
        *   \returns The CommandReply after execution
        */
        CommandReply _run(Command& cmd);

        /*!
        *   \brief Execute a list of commands
        *   \param cmds The CommandList to execute
        *   \returns The CommandReply from the last
        *            command execution
        */
        CommandReply _run(CommandList& cmd_list);

        /*!
        *  \brief Set the prefixes that are used for
        *         set and get methods using SSKEYIN and
        *         SSKEYOUT environment variables.
        */
        void _set_prefixes_from_env();

        /*!
        *  \brief Get the key prefix for placement methods
        *  \returns std::string container the placement prefix
        */
        inline std::string _put_prefix();

        /*!
        *  \brief Get the key prefix for retrieval methods
        *  \returns std::string container the retrieval prefix
        */
        inline std::string _get_prefix();

        /*!
        *  \brief Append a vector of keys with the retrieval prefix
        *  \param keys The vector of keys to prefix for retrieval
        */
        inline void _append_with_get_prefix(std::vector<std::string>& keys);

        /*!
        *  \brief Append a vector of keys with the placement prefix
        *  \param keys The vector of keys to prefix for placement
        */
        inline void _append_with_put_prefix(std::vector<std::string>& keys);

        /*!
        *  \brief Execute the command to retrieve the DataSet metadata portion
        *         of the DataSet.
        *   \param name The name of the DataSet, not prefixed.
        *   \returns The CommandReply for the command to retrieve the DataSet
        *             metadata
        */
        inline CommandReply _get_dataset_metadata(const std::string& name);

        /*!
        *   \brief Retrieve the tensor from the DataSet and return
        *          a TensorBase object that can be used to return
        *          tensor information to the user.  The returned
        *          TensorBase object has been dynamically allocated,
        *          but not yet tracked for memory management in
        *          any object.
        *   \details The TensorBase object returned will always
        *            have a MemoryLayout::contiguous layout.
        *   \param name  The name used to reference the tensor
        *   \returns A TensorBase object.
        */
        TensorBase* _get_tensorbase_obj(const std::string& name);

        friend class PyClient;

    private:

        /*!
        *  \brief SharedMemoryList to manage memory associated
        *         with model retrieval requests
        */
        SharedMemoryList<char> _model_queries;

        /*!
        *  \brief SharedMemoryList to manage memory associated
        *         with tensor dimensions from tensor retrieval
        */
        SharedMemoryList<size_t> _dim_queries;

        /*!
        *  \brief The _tensor_pack memory is not for querying
        *         by name, but is used to manage memory.
        */
        TensorPack _tensor_memory;

        /*!
        *  \brief The prefix for keys during placement
        */
        std::string _put_key_prefix;

        /*!
        *  \brief The prefix for keys during retrieval
        */
        std::string _get_key_prefix;

        /*!
        *  \brief Vector of all potential retrieval prefixes
        */
        std::vector<std::string> _get_key_prefixes;

        /*!
        * \brief Flag determining whether prefixes should be used
        *        for tensor keys.
        */
        bool _use_tensor_prefix;

        /*!
        * \brief Flag determining whether prefixes should be used
        *        for model and script keys.
        */
        bool _use_model_prefix;

        /*!
        * \brief Build full formatted key of a tensor, based on current prefix settings.
        * \param key Tensor key
        * \param on_db Indicates whether the key refers to an entity which is already in the database.
        */
        inline std::string _build_tensor_key(const std::string& key, bool on_db);

        /*!
        * \brief Build full formatted key of a model or a script, based on current prefix settings.
        * \param key Model or script key
        * \param on_db Indicates whether the key refers to an entity which is already in the database.
        */
        inline std::string _build_model_key(const std::string& key, bool on_db);

        /*!
        *  \brief Build full formatted key of a dataset, based
        *         on current prefix settings.
        *  \param dataset_name Name of dataset
        *  \param on_db Indicates whether the key refers to an entity which is already in the database.
        *  \returns Formatted key.
        */
        inline std::string _build_dataset_key(const std::string& dataset_name, bool on_db);

        /*!
        *  \brief Create the key for putting or getting a DataSet tensor in the database
        *  \param dataset_name The name of the dataset
        *  \param tensor_name The name of the tensor
        *  \param on_db Indicates whether the key refers to an entity which is already in the database.
        *  \returns A string of the key for the tensor
        */
        inline std::string _build_dataset_tensor_key(const std::string& dataset_name,
                                                     const std::string& tensor_name,
                                                     bool on_db);

        /*!
        *  \brief Create the key for putting or getting DataSet metadata in the database
        *  \param dataset_name The name of the dataset
        *  \param on_db Indicates whether the key refers to an entity which is already in the database.
        *  \returns A string of the key for the metadata
        */
        inline std::string _build_dataset_meta_key(const std::string& dataset_name,
                                                   bool on_db);

        /*!
        *  \brief Create the key to place an indicator in the database that the dataset has been
        *         successfully stored.
        *  \param dataset_name The name of the dataset
        *  \param on_db Indicates whether the key refers to an entity which is already in the database.
        *  \returns A string of the key for the ack key
        */
        inline std::string _build_dataset_ack_key(const std::string& dataset_name,
                                                  bool on_db);

        /*!
        *   \brief Append the Command associated with
        *          placing DataSet metadata in the database
        *          to a CommandList
        *   \param cmd_list The CommandList to append DataSet
        *                   metadata commands
        *   \param dataset The dataset used for the Command
        *                  construction
        */
        void _append_dataset_metadata_commands(CommandList& cmd_list,
                                               DataSet& dataset);

        /*!
        *   \brief Append the Command associated with
        *          placing DataSet tensors in the database
        *          to a CommandList
        *   \param cmd_list The CommandList to append DataSet
        *                   tensor commands
        *   \param dataset The dataset used for the Command
        *                  construction
        */
        void _append_dataset_tensor_commands(CommandList& cmd_list,
                                             DataSet& dataset);

        /*!
        *   \brief Append the Command associated with
        *          acknowledging that the DataSet is complete
        *          (all put commands processed) to the CommandList
        *   \param cmd_list The CommandList to append DataSet
        *                   ack command
        *   \param dataset The dataset used for the Command
        *                  construction
        */
        void _append_dataset_ack_command(CommandList& cmd_list,
                                         DataSet& dataset);

        /*!
        *   \brief Put the metadata fields embedded in a
        *          CommandReply into the DataSet
        *   \param dataset The DataSet that will have
        *                  metadata place in it.
        *   \param reply The CommandReply containing the
        *                metadata fields.
        */
        void _unpack_dataset_metadata(DataSet& dataset,
                                      CommandReply& reply);
};

} //namespace SmartRedis

#endif //__cplusplus
#endif //SMARTREDIS_CPP_CLIENT_H
