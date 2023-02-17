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

#ifndef SMARTREDIS_REDISCLUSTER_H
#define SMARTREDIS_REDISCLUSTER_H

#include <unordered_set>
#include <mutex>

#include "redisserver.h"
#include "dbnode.h"
#include "nonkeyedcommand.h"
#include "keyedcommand.h"
#include "pipelinereply.h"
#include "address.h"

namespace SmartRedis {

///@file

class SRObject;

/*!
*   \brief  The RedisCluster class executes RedisServer
*           commands on a redis cluster.
*/
class RedisCluster : public RedisServer
{
    public:

        /*!
        *   \brief RedisCluster constructor.
        *   \param context The owning context
        */
        RedisCluster(const SRObject* context);

        /*!
        *   \brief RedisCluster constructor.
        *          Uses address provided to constructor instead
        *          of environment variables.
        *   \param context The owning context
        *   \param address_spec The TCP or UDS address of the server
        */
        RedisCluster(const SRObject* context, std::string address_spec);

        /*!
        *   \brief RedisCluster copy constructor is not allowed
        *   \param cluster The RedisCluster to copy for construction
        */
        RedisCluster(const RedisCluster& cluster) = delete;

        /*!
        *   \brief RedisCluster copy assignment is not allowed
        *   \param cluster The RedisCluster to copy for assignment
        */
        RedisCluster& operator=(const RedisCluster& cluster) = delete;

        /*!
        *   \brief RedisCluster destructor
        */
        ~RedisCluster();

        /*!
        *   \brief RedisCluster move constructor
        *   \param cluster The RedisCluster to move for construction
        */
        RedisCluster(RedisCluster&& cluster) = default;

        /*!
        *   \brief RedisCluster move assignment
        *   \param cluster The RedisCluster to move for assignment
        */
        RedisCluster& operator=(RedisCluster&& cluster) = default;

        /*!
        *   \brief Run a single-key Command on the server
        *   \param cmd The single-key Comand to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(SingleKeyCommand& cmd);

        /*!
        *   \brief Run a multi-key Command on the server
        *   \param cmd The multi-key Comand to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(MultiKeyCommand& cmd);

        /*!
        *   \brief Run a compound Command on the server
        *   \param cmd The compound Comand to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(CompoundCommand& cmd);

        /*!
        *   \brief Run a non-keyed Command that
        *          addresses the given db node on the server
        *   \param cmd The non-keyed Command that
        *              addresses the given db node
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(AddressAtCommand& cmd);

        /*!
        *   \brief Run a non-keyed Command that
        *          addresses any db node on the server
        *   \param cmd The non-keyed Command that
        *              addresses any db node
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(AddressAnyCommand& cmd);

        /*!
        *   \brief Run a non-keyed Command that
        *          addresses every db node on the server
        *   \param cmd The non-keyed Command that
        *              addresses any db node
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(AddressAllCommand& cmd);

        /*!
        *   \brief Run multiple single-key or single-hash slot
        *          Command on the server.  Each Command in the
        *          CommandList is run sequentially.
        *   \param cmd The CommandList containing multiple
        *              single-key or single-hash
        *              slot Command to run
        *   \returns A list of CommandReply for each Command
        *            in the CommandList
        */
        virtual std::vector<CommandReply> run(CommandList& cmd);

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
        run_via_unordered_pipelines(CommandList& cmd_list);

        /*!
        *   \brief Check if a key exists in the database. This
        *          function does not work for models and scripts.
        *          For models and scripts, model_key_exists should
        *          be used.
        *   \param key The key to check
        *   \returns True if the key exists, otherwise False
        */
        virtual bool key_exists(const std::string& key);

        /*!
        *   \brief Check if a hash field exists
        *   \param key The key containing the field
        *   \param field The field in the key to check
        *   \returns True if the hash field exists, otherwise False
        */
        virtual bool hash_field_exists(const std::string& key,
                                       const std::string& field);

        /*!
        *   \brief Check if a model or script key exists in the database
        *   \param key The key to check
        *   \returns True if the key exists, otherwise False
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
        *   \returns The CommandReply from the put tensor
        *            command execution
        */
        virtual CommandReply put_tensor(TensorBase& tensor);

        /*!
        *   \brief Get a Tensor from the server
        *   \param key The name of the tensor to retrieve
        *   \returns The CommandReply from the get tensor server
        *            command execution
        */
        virtual CommandReply get_tensor(const std::string& key);

        /*!
        *   \brief Rename a tensor in the database
        *   \param key The original key for the tensor
        *   \param new_key The new key for the tensor
        *   \returns The CommandReply from the last Command
        *            execution in the renaming of the tensor.
        *            In the case of RedisCluster, this is
        *            the reply for the final delete_tensor call.
        */
        virtual CommandReply rename_tensor(const std::string& key,
                                           const std::string& new_key);

        /*!
        *   \brief Delete a tensor in the database
        *   \param key The database key for the tensor
        *   \returns The CommandReply from delete command
        *            executed on the server
        */
        virtual CommandReply delete_tensor(const std::string& key);

        /*!
        *   \brief Copy a tensor from the source key to
        *          the destination key
        *   \param src_key The source key for the tensor copy
        *   \param dest_key The destination key for the tensor copy
        *   \returns The CommandReply from the last Command
        *            execution in the copying of the tensor.
        *            In the case of RedisCluster, this is
        *            the CommandReply from a put_tensor commands.
        */
        virtual CommandReply copy_tensor(const std::string& src_key,
                                         const std::string& dest_key);

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
                                          const std::vector<std::string>& dest);

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
                                            = std::vector<std::string>());

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
                                            = std::vector<std::string>());

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
        *   \param inputs The keys of inputs tensors to use
        *                 in the model
        *   \param outputs The keys of output tensors that
        *                 will be used to save model results
        *   \returns The CommandReply from the run model server
        *            Command
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
        *   \param inputs The keys of inputs tensors to use
        *                 in the script
        *   \param outputs The keys of output tensors that
        *                 will be used to save script results
        *   \returns The CommandReply from script run Command
        *            execution
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
        *   \returns The CommandReply from script delete Command execution
        *   \throw SmartRedis::Exception if model deletion fails
        */
        CommandReply delete_model(const std::string& key);

        /*!
        *   \brief Remove a model from the database that was stored
        *          for use with multiple GPUs
        *   \param name The name associated with the model
        *   \param first_cpu the first GPU (zero-based) to use with the model
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
        CommandReply delete_script(const std::string& key);

        /*!
        *   \brief Remove a script from the database that was stored
        *          for use with multiple GPUs
        *   \param name The name associated with the script
        *   \param first_cpu the first GPU (zero-based) to use with the script
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
        */
        virtual CommandReply get_model(const std::string& key);

        /*!
        *   \brief Retrieve the script from the database
        *   \param key The key associated with the script
        *   \returns The CommandReply that contains the result
        *            of the get script execution on the server
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
        */
        virtual CommandReply
        get_model_script_ai_info(const std::string& address,
                                 const std::string& key,
                                 const bool reset_stat);

    protected:

        /*!
        *   \brief Get a DBNode prefix for the provided hash slot
        *   \param hash_slot The hash slot to get a prefix for
        *   \throw RuntimeException if there is an error
        *          creating a prefix for a particular hash slot
        *          or if hash_slot is greater than 16384.
        */
        std::string _get_crc16_prefix(uint64_t hash_slot);

    private:

        /*!
        *   \brief sw::redis::RedisCluster object pointer
        */
        sw::redis::RedisCluster* _redis_cluster;

        /*!
        *   \brief Vector of DBNodes in the cluster
        */
        std::vector<DBNode> _db_nodes;

        /*!
        *   \brief Prefix of the most recently used DBNode
        */
        std::string _last_prefix;

        /*!
        *   \brief Run the command on the correct db node
        *   \param cmd The command to run on the server
        *   \param db_prefix The prefix of the db node the
        *                    command addresses
        *   \returns The CommandReply from the
        *            command execution
        */
        inline CommandReply _run(const Command& cmd, std::string db_prefix);

        /*!
        *   \brief Connect to the cluster at the address and port
        *   \param address_port A string formatted as
        *                       tcp:://address:port
        *                       for redis connection
        */
        inline void _connect(SRAddress& db_address);

        /*!
        *   \brief Map the RedisCluster via the CLUSTER SLOTS
        *          command.
        */
        inline void _map_cluster();

        /*!
        *   \brief Get the prefix that can be used to address
        *          the correct database for a given command
        *   \param cmd The Command to analyze for DBNode prefix
        *   \returns The DBNode prefix as a string
        *   \throw RuntimeException if the Command does not have
        *          keys or if multiple keys have different prefixes
        */
        std::string _get_db_node_prefix(Command& cmd);

        /*!
        *   \brief Get the index of the database node corresponding
        *          to the provided key
        *   \param key The command key
        *   \returns The index in _db_nodes corresponding to the key
        */
        inline uint16_t _get_db_node_index(const std::string& key);

        /*!
        *   \brief Processes the CommandReply for CLUSTER SLOTS
        *          to build DBNode information
        *   \param reply The CommandReply for CLUSTER SLOTS
        *   \throw RuntimeException if there is an error
        *          creating a prefix for a particular DBNode
        */
        inline void _parse_reply_for_slots(CommandReply& reply);

        /*!
        *   \brief Perfrom an inverse XOR and shift using
        *          the CRC16 polynomial starting at the bit
        *          position specified by initial_shift.
        *          The XOR shift will be performed on n_bits.
        *          The XOR operation is performed on every
        *          non-zero bit starting from the right, following
        *          the same pattern as the forward CRC16 calculation
        *          (which starts on the left).
        *   \param remainder The polynomial expression used
        *          with the CRC16 polynomial.  remainder
        *          is modified and contains the result of the
        *          inverse CRC16 XOR shifts.
        *   \param initial_shift An initial shift applied to
        *          the polynomial so the inverse XOR shift can be
        *          restarted at the initial_shift bit position.
        *   \param n_bits The number of bits (e.g. the number of
        *                 times) the XOR operation should be
        *                 applied to remainder
        */
        void _crc_xor_shift(uint64_t& remainder,
                            const size_t initial_shift,
                            const size_t n_bits);

        /*!
        *   \brief Check if the current CRC16 inverse
        *          contains any forbidden characters
        *   \param char_bits The character bits (current solution)
        *   \param n_chars The current number of characters
        *   \return True if the char_bits contain a forbidden
        *           character, otherwise False.
        */
        bool _is_valid_inverse(uint64_t char_bits,
                               const size_t n_chars);

        /*!
        *   \brief Determine if the key has a substring
        *          enclosed by "{" and "}" characters
        *   \param key The key to examine
        *   \returns True if the key contains a substring
        *            enclosed by "{" and "}" characters
        */
        bool _has_hash_tag(const std::string& key);

        /*!
        *   \brief  Return the key enclosed by "{" and "}"
        *           characters
        *   \param key The key to examine
        *   \returns The first substring enclosed by "{" and "}"
        *           characters
        */
        std::string _get_hash_tag(const std::string& key);

        /*!
        *   \brief  Get the hash slot for a key
        *   \param key The key to examine
        *   \returns The hash slot of the key
        */
        uint16_t _get_hash_slot(const std::string& key);

        /*!
        *   \brief  Get the index of the DBNode responsible
        *           for the hash slot
        *   \param hash_slot The hash slot to search for
        *   \param lhs The lower bound of the DBNode array to
        *              search (inclusive)
        *   \param rhs The upper bound of the DBNode array to
        *              search (inclusive)
        *   \returns DBNode index responsible for the hash slot
        */
        uint16_t _db_node_hash_search(uint16_t hash_slot,
                                      unsigned lhs,
                                      unsigned rhs);

        /*!
        *   \brief  Attaches a prefix and constant suffix to keys to
        *           enforce identical hash slot constraint
        *   \param names The keys that need to be updated for identical
        *                hash slot constraint
        *   \param db_prefix The prefix to attach
        *   \returns A vector of updated names
        */
        std::vector<std::string>  _get_tmp_names(std::vector<std::string> names,
                                                 std::string db_prefix);

        /*!
        *   \brief  Delete multiple keys with the assumption that all
        *           keys use the same hash slot
        *   \param keys The keys to be deleted
        *   \returns A vector of updated names
        */
        void _delete_keys(std::vector<std::string> keys);

        /*!
        *   \brief  Run a model in the database that uses dagrun
        *   \param key The key associated with the model
        *   \param inputs The keys of inputs tensors to use
        *                 in the model
        *   \param outputs The keys of output tensors that
        *                 will be used to save model results
        */
        void __run_model_dagrun(const std::string& key,
                                std::vector<std::string> inputs,
                                std::vector<std::string> outputs);

        /*!
        *   \brief  Retrieve the optimum model prefix for
        *           the set of inputs
        *   \param name The name of the model
        *   \param inputs The keys of inputs tensors to use
        *                 in the model
        *   \param outputs The keys of output tensors that
        *                 will be used to save model results
        */
        DBNode* _get_model_script_db(const std::string& name,
                                     std::vector<std::string>& inputs,
                                     std::vector<std::string>& outputs);

        /*!
        *   \brief Execute a pipeline for the provided commands.
        *          The provided commands MUST be executable on a single
        *          shard.
        *   \param cmds Vector of Command pointers to execute
        *   \param shard_prefix The prefix corresponding to the shard
        *                       where the pipeline is executed
        *   \throw SmartRedis::Exception if an error is encountered following
        *          multiple attempts
        *   \return A PipelineReply for the provided commands.  The
        *           PipelineReply will be in the same order as the provided
        *           Command vector.
        */
        PipelineReply _run_pipeline(std::vector<Command*>& cmds,
                                    std::string& shard_prefix);
};

} // namespace SmartRedis

#endif // SMARTREDIS_REDISCLUSTER_H
