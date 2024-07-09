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

#ifndef SMARTREDIS_REDIS_H
#define SMARTREDIS_REDIS_H

#include "redisserver.h"

///@file

namespace SmartRedis {

class ConfigOptions;

/*!
*   \brief  The Redis class executes RedisServer
*           commands on a non-cluster redis server.
*/
class Redis : public RedisServer
{
    public:
        /*!
        *   \brief Redis constructor.
        *   \param cfgopts Our source for configuration options
        */
        Redis(ConfigOptions* cfgopts);

        /*!
        *   \brief Redis constructor.
        *          Uses address provided to constructor instead
        *          of environment variables.
        *   \param cfgopts Our source for configuration options
        *   \param addr_spec The TCP or UDS server address
        *   \throw SmartRedis::Exception if connection fails
        */
        Redis(ConfigOptions* cfgopts, std::string addr_spec);

        /*!
        *   \brief Redis copy constructor is not allowed
        *   \param cluster The Redis to copy for construction
        */
        Redis(const Redis& cluster) = delete;

        /*!
        *   \brief Redis copy assignment is not allowed
        *   \param cluster The Redis to copy for assignment
        */
        Redis& operator=(const Redis& cluster) = delete;

        /*!
        *   \brief Redis destructor
        */
        ~Redis();

        /*!
        *   \brief Redis move constructor
        *   \param cluster The Redis to move for construction
        */
        Redis(Redis&& cluster) = default;

        /*!
        *   \brief Redis move assignment
        *   \param cluster The Redis to move for assignment
        */
        Redis& operator=(Redis&& cluster) = default;

       /*!
        *   \brief Run a SingleKeyCommand on the server
        *   \param cmd The SingleKeyCommand to run
        *   \returns The CommandReply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual CommandReply run(SingleKeyCommand& cmd);

        /*!
        *   \brief Run a MultiKeyCommand on the server
        *   \param cmd The MultiKeyCommand to run
        *   \returns The CommandReply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual CommandReply run(MultiKeyCommand& cmd);

        /*!
        *   \brief Run a CompoundCommand on the server
        *   \param cmd The CompoundCommand to run
        *   \returns The CommandReply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual CommandReply run(CompoundCommand& cmd);

        /*!
        *   \brief Run an AddressAtCommand on the server
        *   \param cmd The AddressAtCommand command to run
        *   \returns The CommandReply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual CommandReply run(AddressAtCommand& cmd);

        /*!
        *   \brief Run an AddressAnyCommand on the server
        *   \param cmd The AddressAnyCommand to run
        *   \returns The CommandReply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual CommandReply run(AddressAnyCommand& cmd);

        /*!
        *   \brief Run a non-keyed Command that
        *          addresses every db node on the server
        *   \param cmd The non-keyed Command that addresses any db node
        *   \returns The CommandReply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual CommandReply run(AddressAllCommand& cmd);

        /*!
        *   \brief Run multiple single-key or single-hash slot
        *          Command on the server.  Each Command in the
        *          CommandList is run sequentially.
        *   \param cmd The CommandList containing multiple single-key or
        *              single-hash slot Command to run
        *   \returns A list of CommandReply for each Command
        *            in the CommandList
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual std::vector<CommandReply> run(CommandList& cmd);

        /*!
        *   \brief Run multiple single-key or single-hash slot
        *          Command on the server in pipelines.  The
        *          Command in the CommandList will be grouped
        *          by shard, and executed in groups by shard.
        *          Commands are not guaranteed to be executed
        *          in any sequence or ordering.
        *   \param cmd_list The CommandList containing multiple single-key
        *                   or single-hash slot Commands to run
        *   \returns A list of CommandReply for each Command
        *            in the CommandList. The order of the result
        *            matches the order of the input CommandList.
        *   \throw SmartRedis::Exception if command execution fails
        */
        virtual PipelineReply
        run_via_unordered_pipelines(CommandList& cmd_list);

        /*!
        *   \brief Check if a key exists in the database. This
        *          function does not work for models and scripts.
        *          For models and scripts, model_key_exists should
        *          be used.
        *   \param key The key to check
        *   \returns True if the key exists, otherwise False
        *   \throw SmartRedis::Exception if existence check fails
        */
        virtual bool key_exists(const std::string& key);

        /*!
        *   \brief Check if a hash field exists
        *   \param key The key containing the field
        *   \param field The field in the key to check
        *   \returns True if the hash field exists, otherwise False
        *   \throw SmartRedis::Exception if existence check fails
        */
        virtual bool hash_field_exists(const std::string& key,
                                       const std::string& field);

        /*!
        *   \brief Check if a model or script key exists in the database
        *   \param key The key to check
        *   \returns True if the key exists, otherwise False
        *   \throw SmartRedis::Exception if existence check fails
        */
        virtual bool model_key_exists(const std::string& key);

        /*!
         *  \brief Check if address is valid
         *  \param address Address (TCP or UDS) of database
         *  \return True if address is valid
         */
        virtual bool is_addressable(const SRAddress& address) const;

        /*!
        *   \brief Put a Tensor on the server
        *   \param tensor The Tensor to put on the server
        *   \returns The CommandReply from the put tensor command execution
        *   \throw SmartRedis::Exception if tensor storage fails
        */
        virtual CommandReply put_tensor(TensorBase& tensor);

        /*!
        *   \brief Get a Tensor from the server
        *   \param key The name of the tensor to retrieve
        *   \returns The CommandReply from the get tensor server
        *            command execution
        *   \throw SmartRedis::Exception if tensor retrieval fails
        */
        virtual CommandReply get_tensor(const std::string& key);

        /*!
        *   \brief Get a list of Tensor from the server
        *   \param keys The keys of the tensor to retrieve
        *   \returns The PipelineReply from executing the get tensor commands
        *   \throw SmartRedis::Exception if tensor retrieval fails
        */
        virtual PipelineReply get_tensors(
            const std::vector<std::string>& keys);

        /*!
        *   \brief Rename a tensor in the database
        *   \param key The original key for the tensor
        *   \param new_key The new key for the tensor
        *   \returns The CommandReply from executing the RENAME command
        *   \throw SmartRedis::Exception if tensor rename fails
        */
        virtual CommandReply rename_tensor(const std::string& key,
                                           const std::string& new_key);

        /*!
        *   \brief Delete a tensor in the database
        *   \param key The database key for the tensor
        *   \returns The CommandReply from delete command
        *            executed on the server
        *   \throw SmartRedis::Exception if tensor removal fails
        */
        virtual CommandReply delete_tensor(const std::string& key);

        /*!
        *   \brief Copy a tensor from the source key to
        *          the destination key
        *   \param src_key The source key for the tensor copy
        *   \param dest_key The destination key for the tensor copy
        *   \returns The CommandReply from executing the COPY command
        *   \throw SmartRedis::Exception if tensor copy fails
        */
        virtual CommandReply copy_tensor(const std::string& src_key,
                                         const std::string& dest_key);

        /*!
        *   \brief Copy a vector of tensors from source keys
        *          to destination keys
        *   \param src Vector of source keys
        *   \param dest Vector of destination keys
        *   \returns The CommandReply from the last put command
        *            associated with the tensor copy
        *   \throw SmartRedis::Exception if tensor copy fails
        */
        virtual CommandReply copy_tensors(const std::vector<std::string>& src,
                                          const std::vector<std::string>& dest);


        /*!
        *   \brief Set a model from std::string_view buffer in the
        *          database for future execution
        *   \param key The key to associate with the model
        *   \param model The model as a sequence of buffer string_view chunks
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model execution
        *   \param min_batch_timeout Max time (ms) to wait for min batch size
        *   \param tag A tag to attach to the model for information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        *   \returns The CommandReply from the set_model Command
        *   \throw RuntimeException for all client errors
        */
        virtual CommandReply set_model(const std::string& key,
                                       const std::vector<std::string_view>& model,
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
        *   \param model The model as a sequence of buffer string_view chunks
        *   \param backend The name of the backend
        *                  (TF, TFLITE, TORCH, ONNX)
        *   \param first_gpu The first GPU to use with this model
        *   \param num_gpus The number of GPUs to use with this model
        *   \param batch_size The batch size for model execution
        *   \param min_batch_size The minimum batch size for model execution
        *   \param min_batch_timeout Max time (ms) to wait for min batch size
        *   \param tag A tag to attach to the model for information purposes
        *   \param inputs One or more names of model input nodes
        *                 (TF models only)
        *   \param outputs One or more names of model output nodes
        *                 (TF models only)
        *   \throw RuntimeException for all client errors
        */
        virtual void set_model_multigpu(const std::string& name,
                                        const std::vector<std::string_view>& model,
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
        *   \brief Set a script from std::string_view buffer in the
        *          database for future execution
        *   \param key The key to associate with the script
        *   \param device The name of the device for execution
        *                 (e.g. CPU or GPU)
        *   \param script The script source in a std::string_view
        *   \returns The CommandReply from set_script Command
        *   \throw RuntimeException for all client errors
        */
        virtual CommandReply set_script(const std::string& key,
                                        const std::string& device,
                                        std::string_view script);

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
                                         int num_gpus);

        /*!
        *   \brief Run a model in the database using the
        *          specified input and output tensors
        *   \param key The key associated with the model
        *   \param inputs The keys of inputs tensors to use in the model
        *   \param outputs The keys of output tensors that
        *                 will be used to save model results
        *   \returns The CommandReply from the run model server Command
        *   \throw RuntimeException for all client errors
        */
        virtual CommandReply run_model(const std::string& key,
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
        *   \param first_gpu The first GPU to use with this model
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw RuntimeException for all client errors
        */
        virtual void run_model_multigpu(const std::string& name,
                                        std::vector<std::string> inputs,
                                        std::vector<std::string> outputs,
                                        int offset,
                                        int first_gpu,
                                        int num_gpus);

        /*!
        *   \brief Run a script function in the database using the
        *          specified input and output tensors
        *   \param key The key associated with the script
        *   \param function The name of the function in the script to run
        *   \param inputs The keys of inputs tensors to use in the script
        *   \param outputs The keys of output tensors that
        *                 will be used to save script results
        *   \returns The CommandReply from script run Command execution
        *   \throw RuntimeException for all client errors
        */
        virtual CommandReply run_script(const std::string& key,
                                        const std::string& function,
                                        std::vector<std::string> inputs,
                                        std::vector<std::string> outputs);

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
                                         int num_gpus);

        /*!
        *   \brief Remove a model from the database
        *   \param key The key associated with the model
        *   \returns The CommandReply from model delete Command execution
        *   \throw SmartRedis::Exception if model deletion fails
        */
        virtual CommandReply delete_model(const std::string& key);

        /*!
        *   \brief Remove a model from the database that was stored
        *          for use with multiple GPUs
        *   \param name The name associated with the model
        *   \param first_gpu the first GPU (zero-based) to use with the model
        *   \param num_gpus the number of gpus for which the model was stored
        *   \throw SmartRedis::Exception if model deletion fails
        */
        virtual void delete_model_multigpu(
            const std::string& name, int first_gpu, int num_gpus);

        /*!
        *   \brief Remove a script from the database
        *   \param key The key associated with the script
        *   \returns The CommandReply from script delete Command execution
        *   \throw SmartRedis::Exception if script deletion fails
        */
        virtual CommandReply delete_script(const std::string& key);

        /*!
        *   \brief Remove a script from the database that was stored
        *          for use with multiple GPUs
        *   \param name The name associated with the script
        *   \param first_gpu the first GPU (zero-based) to use with the script
        *   \param num_gpus the number of gpus for which the script was stored
        *   \throw SmartRedis::Exception if script deletion fails
        */
        virtual void delete_script_multigpu(
            const std::string& name, int first_gpu, int num_gpus);

        /*!
        *   \brief Retrieve the model from the database
        *   \param key The key associated with the model
        *   \returns The CommandReply that contains the result
        *            of the get model execution on the server
        *   \throw SmartRedis::Exception if model retrieval fails
        */
        virtual CommandReply get_model(const std::string& key);

        /*!
        *   \brief Retrieve the script from the database
        *   \param key The key associated with the script
        *   \returns The CommandReply that contains the result
        *            of the get script execution on the server
        *   \throw SmartRedis::Exception if script retrieval fails
        */
        virtual CommandReply get_script(const std::string& key);

        /*!
        *   \brief Retrieve model/script runtime statistics
        *   \param address The address of the database node (host:port)
        *   \param key The key associated with the model or script
        *   \param reset_stat Boolean indicating if the counters associated
        *                     with the model or script should be reset.
        *   \returns The CommandReply that contains the result
        *            of the AI.INFO execution on the server
        *   \throw SmartRedis::Exception if info retrieval fails
        */
        virtual CommandReply
        get_model_script_ai_info(const std::string& address,
                                 const std::string& key,
                                 const bool reset_stat);

        /*!
        *   \brief Retrieve the current model chunk size
        *   \returns The size in bytes for model chunking
        */
        virtual int get_model_chunk_size();

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
        virtual void set_model_chunk_size(int chunk_size);

        /*!
        *   \brief Run a CommandList via a Pipeline
        *   \param cmdlist The list of commands to run
        *   \returns The PipelineReply with the result of command execution
        *   \throw SmartRedis::Exception if execution fails
        */
        PipelineReply run_in_pipeline(CommandList& cmdlist);

        /*!
        *   \brief Create a string representation of the Redis connection
        *   \returns A string representation of the Redis connection
        */
        virtual std::string to_string() const;

    private:

        /*!
        *   \brief sw::redis::Redis object pointer
        */
        sw::redis::Redis* _redis;

        /*!
        *   \brief Run a Command on the server
        *   \param cmd The Command to run
        *   \returns The CommandReply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        inline CommandReply _run(const Command& cmd);

        /*!
        *   \brief Inserts an address into _address_node_map
        *   \param db_address The server address
        */
        inline void _add_to_address_map(SRAddress& db_address);

        /*!
        *   \brief Connect to the server at the address and port
        *   \param db_address The server address
        *   \throw SmartRedis::Exception if connection fails
        */
        inline void _connect(SRAddress& db_address);

        /*!
        *   \brief Pipeline execute a series of commands
        *   \param cmds The commands to execute
        *   \returns Pipeline reply from the command execution
        *   \throw SmartRedis::Exception if command execution fails
        */
        PipelineReply _run_pipeline(std::vector<Command*>& cmds);

};

} // namespace SmartRedis

#endif // SMARTREDIS_REDIS_H
