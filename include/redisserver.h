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

#ifndef SMARTREDIS_CPP_REDISSERVER_H
#define SMARTREDIS_CPP_REDISSERVER_H

#include <thread>
#include <iostream>
#include <sw/redis++/redis++.h>
#include "command.h"
#include "commandreply.h"
#include "commandreplyparser.h"
#include "commandlist.h"
#include "tensorbase.h"
#include "dbnode.h"

///@file

namespace SmartRedis {

class RedisServer;


/*!
*   \brief Abstract class that defines interface for
*          objects that execute commands on server.
*/
class RedisServer {

    public:

        /*!
        *   \brief Default constructor
        */
        RedisServer() = default;

        /*!
        *   \brief Default destructor
        */
        virtual ~RedisServer() = default;

        /*!
        *   \brief Run a single-key or single-hash slot
        *          Command on the server
        *   \param cmd The single-key or single-hash
        *              slot Comand to run
        *   \returns The CommandReply from the
        *            command execution
        */
        virtual CommandReply run(Command& cmd) = 0;

        /*!
        *   \brief Run multiple single-key or single-hash slot
        *          Command on the server.  Each Command in the
        *          CommandList is run sequentially.
        *   \param cmd The CommandList containing multiple
        *              single-key or single-hash
        *              slot Comand to run
        *   \returns The CommandReply from the last
        *            command execution
        */
        virtual CommandReply run(CommandList& cmd) = 0;

        /*!
        *   \brief Check if a key exists in the database
        *   \param key The key to check
        *   \returns True if the key exists, otherwise False
        */
        virtual bool key_exists(const std::string& key) = 0;

        /*!
         *  \brief Check if a model or script exists in the database
         *  \param key The script or model key
         *  \return True if the model or script exists
         */
        virtual bool model_key_exists(const std::string& key) = 0;

        /*!
         *  \brief Check if address and port maps to database node
         *  \param addresss address of database
         *  \param port port of database
         *  \return True if address is valid
         */
        virtual bool is_addressable(const std::string& address, const uint64_t& port) = 0;

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
        *   \param src_key Vector of source keys
        *   \param dest_key Vector of destination keys
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
        *   \brief Run a model in the database using the
        *          specificed input and output tensors
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
        *   \brief Run a script function in the database using the
        *          specificed input and output tensors
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

    protected:

        /*!
        *   \brief Retrieve a single address, randomly
        *          chosen from a list of addresses if
        *          applicable, from the SSDB environment
        *          variable.
        *   \returns A address and port pair in the form of
        *            address:port
        */
        std::string _get_ssdb();

        /*!
        *   \brief Unordered map of address:port to DBNode in the cluster
        */
        std::unordered_map<std::string, DBNode*> _address_node_map;

        /*!
        *   \brief Check that the SSDB environment variable
        *          value does not have any errors
        *   \throw std::runtime_error fi there is an error
        *          in SSDB environment variable format
        */
        void _check_ssdb_string(const std::string& env_str);

};

} // namespace SmartRedis

#endif //SMARTREDIS_CPP_REDISSERVER_H
