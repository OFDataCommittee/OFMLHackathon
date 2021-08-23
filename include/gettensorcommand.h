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

#ifndef SMARTREDIS_GETTENSORCOMMAND_H
#define SMARTREDIS_GETTENSORCOMMAND_H

#include "singlekeycommand.h"
#include "tensorbase.h"
#include "enums/cpp_tensor_type.h"

///@file

namespace SmartRedis {

class RedisServer;

class GetTensorCommand : public SingleKeyCommand
{
    public:
        /*!
        *   \brief Run this Command on the RedisServer.
        *   \param server A pointer to the RedisServer
        */
        virtual CommandReply run_me(RedisServer* server);

        /*!
        *   \brief This function will fill a vector with the dimensions of
        *          the tensor.
        *   \details We assume right now that the META reply is always in the
        *            same order and we can index base reply elements array
        *   \param reply CommandReply from running "AI.TENSORGET"
        *   \return std::vecotr<size_t> with dimensions of the tensor
        */
        static std::vector<size_t> get_dims(CommandReply& reply);

        /*!
        *   \brief This function returns a string view of the data
        *          tensor blob value
        *   \details We are going to assume right now that the meta data
        *            reply is always in the same order and we are going
        *            to just index into the base reply.
        *   \param reply CommandReply from running "AI.TENSORGET"
        *   \return string view of the data tensor blob value
        */
        static std::string_view get_data_blob(CommandReply& reply);

        /*!
        *   \brief This function returns a string of the tensor type
        *   \details We are going to assume right now that the meta
        *            data reply is always in the same order and we
        *            are going to just index into the base reply.
        *   \param reply CommandReply from running "AI.TENSORGET"
        *   \return string of the tensor type
        */
       static TensorType get_data_type(CommandReply& reply);
};

} //namespace SmartRedis

#endif //GETTENSORCOMMAND