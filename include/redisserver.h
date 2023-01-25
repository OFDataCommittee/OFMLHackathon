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

#ifndef SMARTREDIS_REDISSERVER_H
#define SMARTREDIS_REDISSERVER_H

#include <thread>
#include <iostream>
#include <random>
#include <limits.h>
#include <sw/redis++/redis++.h>

#include "command.h"
#include "commandreply.h"
#include "commandlist.h"
#include "tensorbase.h"
#include "dbnode.h"
#include "nonkeyedcommand.h"
#include "keyedcommand.h"
#include "multikeycommand.h"
#include "singlekeycommand.h"
#include "compoundcommand.h"
#include "addressatcommand.h"
#include "addressanycommand.h"
#include "addressallcommand.h"
#include "clusterinfocommand.h"
#include "dbinfocommand.h"
#include "gettensorcommand.h"
#include "pipelinereply.h"
#include "threadpool.h"
#include "address.h"

///@file

namespace SmartRedis {

class SRObject;

/*!
*   \brief Abstract class that defines interface for
*          objects that execute commands on server.
*/
class RedisServer {

    public:

        /*!
        *   \brief Default constructor
        *   \param context The owning context
        */
        RedisServer(const SRObject* context);

        /*!
        *   \brief Destructor
        */
        virtual ~RedisServer();

        /*!
        *   \brief Run a single-key Command on the server
        *   \param cmd The single-key Comand to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(SingleKeyCommand& cmd) = 0;

        /*!
        *   \brief Run a multi-key Command on the server
        *   \param cmd The multi-key Comand to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(MultiKeyCommand& cmd) = 0;

        /*!
        *   \brief Run a compound Command on the server
        *   \param cmd The compound Comand to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(CompoundCommand& cmd) = 0;

        /*!
        *   \brief Run a non-keyed Command that
        *          addresses the given db node on the server
        *   \param cmd The non-keyed Command that
        *              addresses the given db node
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(AddressAtCommand& cmd) = 0;

        /*!
        *   \brief Run a non-keyed Command that
        *          addresses any db node on the server
        *   \param cmd The non-keyed Command that
        *              addresses any db node
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(AddressAnyCommand& cmd) = 0;

        /*!
        *   \brief Run a non-keyed Command that
        *          addresses every db node on the server
        *   \param cmd The non-keyed Command that
        *              addresses any db node
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(AddressAllCommand& cmd) = 0;

        /*!
        *   \brief Run multiple single-key or single-hash slot
        *          Command on the server.  Each Command in the
        *          CommandList is run sequentially.
        *   \param cmd The CommandList containing multiple
        *              single-key or single-hash
        *              slot Comand to run
        *   \returns A list of CommandReply for each Command
        *            in the CommandList
        */
        virtual std::vector<CommandReply> run(CommandList& cmd) = 0;

        /*!
        *   \brief Run multiple single-key or single-hash slot
        *          Command on the server in pipelines.  The
        *          Command in the CommandList will be grouped
        *          by shard, and executed in groups by shard.
        *          Commands are not guaranteed to be executed
        *          in any sequence or ordering.
        *   \param cmd The CommandList containing multiple
        *              single-key or single-hash
        *              slot Command to run
        *   \returns A list of CommandReply for each Command
        *            in the CommandList. The order of the result
        *            matches the order of the input CommandList.
        */
        virtual PipelineReply
        run_via_unordered_pipelines(CommandList& cmd_list) = 0;

        /*!
        *   \brief Check if a key exists in the database
        *   \param key The key to check
        *   \returns True if the key exists, otherwise False
        */
        virtual bool key_exists(const std::string& key) = 0;

        /*!
        *   \brief Check if a hash field exists
        *   \param key The key containing the field
        *   \param field The field in the key to check
        *   \returns True if the hash field exists, otherwise False
        */
        virtual bool hash_field_exists(const std::string& key,
                                       const std::string& field) = 0;

        /*!
         *  \brief Check if a model or script exists in the database
         *  \param key The script or model key
         *  \return True if the model or script exists
         */
        virtual bool model_key_exists(const std::string& key) = 0;

        /*!
         *  \brief Check if address is valid
         *  \param address Address (TCP or UDS) of database
         *  \return True if address is valid
         */
        virtual bool is_addressable(const SRAddress& address) const = 0;

        /*!
        *   \brief Put a Tensor on the server
        *   \param tensor The Tensor to put on the server
        *   \returns The CommandReply from the put tensor
        *            command execution
        */
        virtual CommandReply put_tensor(TensorBase& tensor) = 0;

        /*!
        *   \brief Get a Tensor from the server
        *   \param key The name of the tensor to retrieve
        *   \returns The CommandReply from the get tensor server
        *            command execution
        */
        virtual CommandReply get_tensor(const std::string& key) = 0;

        /*!
        *   \brief Rename a tensor in the database
        *   \param key The original key for the tensor
        *   \param new_key The new key for the tensor
        *   \returns The CommandReply from the last Command
        *            execution in the renaming of the tensor.
        *            Different implementations may have different
        *            sequences of commands.
        */
        virtual CommandReply rename_tensor(const std::string& key,
                                           const std::string& new_key)
                                           = 0;

        /*!
        *   \brief Delete a tensor in the database
        *   \param key The database key for the tensor
        *   \returns The CommandReply from delete command
        *            executed on the server
        */
        virtual CommandReply delete_tensor(const std::string& key) = 0;

        /*!
        *   \brief Copy a tensor from the source key to
        *          the destination key
        *   \param src_key The source key for the tensor copy
        *   \param dest_key The destination key for the tensor copy
        *   \returns The CommandReply from the last Command
        *            execution in the copying of the tensor.
        *            Different implementations may have different
        *            sequences of commands.
        */
        virtual CommandReply copy_tensor(const std::string& src_key,
                                         const std::string& dest_key)
                                         = 0;

        /*!
        *   \brief Copy a vector of tensors from source keys
        *          to destination keys
        *   \param src Vector of source keys
        *   \param dest Vector of destination keys
        *   \returns The CommandReply from the last Command
        *            execution in the copying of the tensor.
        *            Different implementations may have different
        *            sequences of commands.
        */
        virtual CommandReply copy_tensors(const std::vector<std::string>& src,
                                          const std::vector<std::string>& dest
                                          ) = 0;

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
        *   \returns The CommandReply from the set_model Command
        */
        virtual CommandReply set_model(const std::string& key,
                                       std::string_view model,
                                       const std::string& backend,
                                       const std::string& device,
                                       int batch_size = 0,
                                       int min_batch_size = 0,
                                       const std::string& tag = "",
                                       const std::vector<std::string>& inputs
                                            = std::vector<std::string>(),
                                       const std::vector<std::string>& outputs
                                            = std::vector<std::string>()
                                       ) = 0;

        /*!
        *   \brief Set a model from std::string_view buffer in the
        *          database for future execution in a multi-GPU system
        *   \param name The name to associate with the model
        *   \param model The model as a continuous buffer string_view
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param first_gpu The first GPU to use with this model
        *   \param num_gpus The number of GPUs to use with this model
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
        virtual void set_model_multigpu(const std::string& name,
                                        const std::string_view& model,
                                        const std::string& backend,
                                        int first_gpu,
                                        int num_gpus,
                                        int batch_size = 0,
                                        int min_batch_size = 0,
                                        const std::string& tag = "",
                                        const std::vector<std::string>& inputs
                                            = std::vector<std::string>(),
                                        const std::vector<std::string>& outputs
                                            = std::vector<std::string>()) = 0;

        /*!
        *   \brief Set a script from std::string_view buffer in the
        *          database for future execution
        *   \param key The key to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script The script source in a std::string_view
        *   \returns The CommandReply from set_script Command
        */
        virtual CommandReply set_script(const std::string& key,
                                        const std::string& device,
                                        std::string_view script) = 0;

        /*!
        *   \brief Set a script from std::string_view buffer in the
        *          database for future execution in a multi-GPU system
        *   \param name The name to associate with the script
        *   \param script The script source in a std::string_view
        *   \param first_gpu The first GPU to use with this script
        *   \param num_gpus The number of GPUs to use with this script
        *   \throw RuntimeException for all client errors
        */
        virtual void set_script_multigpu(const std::string& name,
                                         const std::string_view& script,
                                         int first_gpu,
                                         int num_gpus) = 0;

        /*!
        *   \brief Run a model in the database using the
        *          specified input and output tensors
        *   \param key The key associated with the model
        *   \param inputs The keys of inputs tensors to use
        *                 in the model
        *   \param outputs The keys of output tensors that
        *                 will be used to save model results
        *   \returns The CommandReply from a Command
        *            execution in the model run execution.
        *            Different implementations may have different
        *            sequences of commands.
        */
        virtual CommandReply run_model(const std::string& key,
                                       std::vector<std::string> inputs,
                                       std::vector<std::string> outputs) = 0;

        /*!
        *   \brief Run a model in the database using the
        *          specified input and output tensors in a multi-GPU system
        *   \param name The name associated with the model
        *   \param inputs The names of input tensors to use in the model
        *   \param outputs The names of output tensors that will be used
        *                  to save model results
        *   \param offset index of the current image, such as a processor
        *                   ID or MPI rank
        *   \param first_gpu The first GPU to use with this model
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw RuntimeException for all client errors
        */
        virtual void run_model_multigpu(const std::string& name,
                                        std::vector<std::string> inputs,
                                        std::vector<std::string> outputs,
                                        int offset,
                                        int first_gpu,
                                        int num_gpus) = 0;

        /*!
        *   \brief Run a script function in the database using the
        *          specified input and output tensors
        *   \param key The key associated with the script
        *   \param function The name of the function in the script to run
        *   \param inputs The keys of inputs tensors to use
        *                 in the script
        *   \param outputs The keys of output tensors that
        *                 will be used to save script results
        *   \returns The CommandReply from a Command
        *            execution in the script run execution.
        *            Different implementations may have different
        *            sequences of commands.
        */
        virtual CommandReply run_script(const std::string& key,
                                        const std::string& function,
                                        std::vector<std::string> inputs,
                                        std::vector<std::string> outputs)
                                         = 0;

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
        *   \param first_gpu The first GPU to use with this script
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw RuntimeException for all client errors
        */
        virtual void run_script_multigpu(const std::string& name,
                                         const std::string& function,
                                         std::vector<std::string>& inputs,
                                         std::vector<std::string>& outputs,
                                         int offset,
                                         int first_gpu,
                                         int num_gpus) = 0;

        /*!
        *   \brief Remove a model from the database
        *   \param key The key associated with the model
        *   \returns The CommandReply from script delete Command execution
        *   \throw SmartRedis::Exception if model deletion fails
        */
        virtual CommandReply delete_model(const std::string& key) = 0;

        /*!
        *   \brief Remove a model from the database that was stored
        *          for use with multiple GPUs
        *   \param name The name associated with the model
        *   \param first_cpu the first GPU (zero-based) to use with the model
        *   \param num_gpus the number of gpus for which the model was stored
        *   \throw SmartRedis::Exception if model deletion fails
        */
        virtual void delete_model_multigpu(
            const std::string& name, int first_gpu, int num_gpus) = 0;

        /*!
        *   \brief Remove a script from the database
        *   \param key The key associated with the script
        *   \returns The CommandReply from script delete Command execution
        *   \throw SmartRedis::Exception if script deletion fails
        */
        virtual CommandReply delete_script(const std::string& key) = 0;

        /*!
        *   \brief Remove a script from the database that was stored
        *          for use with multiple GPUs
        *   \param name The name associated with the script
        *   \param first_cpu the first GPU (zero-based) to use with the script
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw SmartRedis::Exception if script deletion fails
        */
        virtual void delete_script_multigpu(
            const std::string& name, int first_gpu, int num_gpus) = 0;

        /*!
        *   \brief Retrieve the model from the database
        *   \param key The key associated with the model
        *   \returns The CommandReply that contains the result
        *            of the get model execution on the server
        */
        virtual CommandReply get_model(const std::string& key) = 0;

        /*!
        *   \brief Retrieve the script from the database
        *   \param key The key associated with the script
        *   \returns The CommandReply that contains the result
        *            of the get script execution on the server
        */
        virtual CommandReply get_script(const std::string& key) = 0;

        /*!
        *   \brief Retrieve model/script runtime statistics
        *   \param address The TCP or UDS address of the database node
        *   \param key The key associated with the model or script
        *   \param reset_stat Boolean indicating if the counters associated
        *                     with the model or script should be reset.
        *   \returns The CommandReply that contains the result
        *            of the AI.INFO execution on the server
        */
        virtual CommandReply
        get_model_script_ai_info(const std::string& address,
                                 const std::string& key,
                                 const bool reset_stat) = 0;

    protected:

        /*!
        *   \brief Timeout (in seconds) of connection attempt(s).
        */
        int _connection_timeout;

        /*!
        *   \brief Timeout (in seconds) of command attempt(s).
        */
        int _command_timeout;

        /*!
        *   \brief Interval (in milliseconds) between connection attempts.
        */
        int _connection_interval;

        /*!
        *   \brief Interval (in milliseconds) between command execution attempts.
        */
        int _command_interval;

        /*!
        *   \brief The number of client connection attempts
        */
        int _connection_attempts;

        /*!
        *   \brief The number of client command execution attempts
        */
        int _command_attempts;

        /*!
        *   \brief Default value of connection timeout (seconds)
        */
        static constexpr int _DEFAULT_CONN_TIMEOUT = 100;

        /*!
        *   \brief Default value of connection attempt intervals (milliseconds)
        */
        static constexpr int _DEFAULT_CONN_INTERVAL = 1000;

        /*!
        *   \brief Default value of command execution timeout (seconds)
        */
        static constexpr int _DEFAULT_CMD_TIMEOUT = 100;

        /*!
        *   \brief Default value of model execution timeout (milliseconds)
        */
        static constexpr int _DEFAULT_MODEL_TIMEOUT = 60 * 1000 * 1000;

        /*!
        *   \brief Default value of command execution attempt
        *          intervals (milliseconds)
        */
        static constexpr int _DEFAULT_CMD_INTERVAL = 1000;

        /*!
        *   \brief Default number of threads for thread pool
        */
        static constexpr int _DEFAULT_THREAD_COUNT = 4;

        /*!
        *   \brief The owning context
        */
        const SRObject* _context;

        /*!
        *   \brief Seeding for the random number engine
        */
        std::random_device _rd;

        /*!
        *   \brief Random number generator
        */
        std::mt19937 _gen;

        /*!
        *   \brief Number of threads for thread pool
        */
        int _thread_count;

        /*!
        *   \brief The thread pool
        */
        ThreadPool *_tp;

        /*
        *   \brief Indicates whether the server was connected to
        *          via a Unix domain socket (true) or TCP connection
        *          (false)
        */
        bool _is_domain_socket;

        /*!
        *   \brief Environment variable for connection timeout
        */
        inline static const std::string _CONN_TIMEOUT_ENV_VAR =
            "SR_CONN_TIMEOUT";

        /*!
        *   \brief Environment variable for connection interval
        */
        inline static const std::string _CONN_INTERVAL_ENV_VAR =
            "SR_CONN_INTERVAL";

        /*!
        *   \brief Environment variable for command execution timeout
        */
        inline static const std::string _CMD_TIMEOUT_ENV_VAR =
            "SR_CMD_TIMEOUT";

        /*!
        *   \brief Environment variable for command execution interval
        */
        inline static const std::string _CMD_INTERVAL_ENV_VAR =
            "SR_CMD_INTERVAL";

        /*!
        *   \brief Environment variable for model execution timeout
        */
        inline static const std::string _MODEL_TIMEOUT_ENV_VAR =
            "SR_MODEL_TIMEOUT";

        /*!
        *   \brief Environment variable for thread count in thread pool
        */
        inline static const std::string _TP_THREAD_COUNT =
            "SR_THREAD_COUNT";

        /*!
        *   \brief Retrieve a single address, randomly
        *          chosen from a list of addresses if
        *          applicable, from the SSDB environment
        *          variable.
        *   \returns An SRAddress representing the selected server address
        */
        SRAddress _get_ssdb();

        /*!
        *   \brief Unordered map of server address string to DBNode in the cluster
        */
        std::unordered_map<std::string, DBNode*> _address_node_map;

        /*!
        *   \brief Check that the SSDB environment variable
        *          value does not have any errors
        *   \throw RuntimeException if there is an error
        *          in SSDB environment variable format
        */
        void _check_ssdb_string(const std::string& env_str);

        /*!
        *   \brief This function checks that _connection_timeout,
        *          _connection_interval, _command_timeout, and
        *          _command_interval, which have been set from environment
        *          variables, are within valid ranges.
        *   \throw SmartRedis::RuntimeException if any of the runtime
        *          settings is outside of the allowable range
        */
        void _check_runtime_variables();

        /*!
        *   \brief Modular arithmetic that supports negative numbers
        *   \param value The number to be modularized
        *   \param modulus the modulus for the operation
        *   \returns value modulo modulus
        */
        int _modulo(int value, int modulus) {
            return (((value % modulus) + modulus) % modulus);
         }

};

} // namespace SmartRedis

#endif // SMARTREDIS_REDISSERVER_H
