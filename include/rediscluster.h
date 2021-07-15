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

#ifndef SMARTREDIS_CPP_CLUSTER_H
#define SMARTREDIS_CPP_CLUSTER_H

#include <unordered_set>
#include "redisserver.h"
#include "dbnode.h"

namespace SmartRedis {

///@file

class RedisCluster;

/*!
*   \brief  The RedisCluster class executes RedisServer
*           commands on a redis cluster.
*/
class RedisCluster : public RedisServer
{
    public:

        /*!
        *   \brief RedisCluster constructor.
        *          Initializes default values but does not connect.
        */
        RedisCluster();

        /*!
        *   \brief RedisCluster constructor.
        *          Uses address provided to constructor instead
        *          of environment variables.
        *   \param address_port The address and port in the form of
        *                       "tcp://address:port"
        */
        RedisCluster(std::string address_port);

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
        *   \brief Run a single-key or single-hash slot
        *          Command on the server
        *   \param cmd The single-key or single-hash
        *              slot Command to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(Command& cmd);

        /*!
        *   \brief Run multiple single-key or single-hash slot
        *          Command on the server.  Each Command in the
        *          CommandList is run sequentially.
        *   \param cmd The CommandList containing multiple
        *              single-key or single-hash
        *              slot Command to run
        *   \returns The CommandReply from the last
        *            command execution
        */
        virtual CommandReply run(CommandList& cmd);

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
        *   \brief Check if a model or script key exists in the database
        *   \param key The key to check
        *   \returns True if the key exists, otherwise False
        */
        virtual bool model_key_exists(const std::string& key);

        /*!
         *  \brief Check if address is valid
         *  \param addresss address of database
         *  \param port port of database
         *  \return True if address is valid
         */
        virtual bool is_addressable(const std::string& address, const uint64_t& port);

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
        *   \param src_key Vector of source keys
        *   \param dest_key Vector of destination keys
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
        *   \brief Run a model in the database using the
        *          specificed input and output tensors
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
        *   \brief Run a script function in the database using the
        *          specificed input and output tensors
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
        *   \brief Connect to the cluster at the address and port
        *   \param address_port A string formatted as
        *                       tcp:://address:port
        *                       for redis connection
        */
        inline void _connect(std::string address_port);

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
        *   \throw std::runtime_error if the Command does not have
        *          keys or if multiple keys have different prefixes
        */
        std::string _get_db_node_prefix(Command& cmd);

        /*!
        *   \brief Processes the CommandReply for CLUSTER SLOTS
        *          to build DBNode information
        *   \param reply The CommandReply for CLUSTER SLOTS
        *   \throw std::runtime_error if there is an error
        *          creating a prefix for a particular DBNode
        */
        inline void _parse_reply_for_slots(CommandReply& reply);

        /*!
        *   \brief Get a DBNode prefix for the provided hash slot
        *   \param hash_slot The hash slot to get a prefix for
        *   \throw std::runtime_error if there is an error
        *          creating a prefix for a particular DBNode
        */
        std::string _get_crc16_prefix(uint64_t hash_slot);

        /*!
        *   \brief Perform an inverse CRC16 calculation.
        *   \details Given a remainder, this function will
        *            calculate a number that when divded by
        *            the CRC16 polynomial, yields the remainder.
        *   \param remainder The polynomial remainder
        *   \returns A 64-bit number that when divided by the
        *            CRC16 polynomial yields the remainder.
        */
        uint64_t _crc16_inverse(uint64_t remainder);

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
        uint16_t _get_dbnode_index(uint16_t hash_slot,
                                   unsigned lhs,
                                   unsigned rhs);

        /*!
        *   \brief  Attaches a prefix and constant suffix to keys to
        *           enforce identical hash slot constraint
        *   \param names The keys that need to be updated for identical
        *                hash slot constraint
        *   \param prefix The prefix to attach
        *   \returns A vector of updated names
        */
        std::vector<std::string>  _get_tmp_names(std::vector<std::string> names,
                                                 std::string db_prefix);

        /*!
        *   \brief  Delete multiple keys with the assumption that all
        *           keys use the same hash slot
        *   \param names The keys that need to be updated for identical
        *                hash slot constraint
        *   \param prefix The prefix to attach
        *   \returns A vector of updated names
        */
        void _delete_keys(std::vector<std::string> key
                          );

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
};

} //namespace SmartRedis

#endif //SMARTREDIS_CPP_CLUSTER_H
