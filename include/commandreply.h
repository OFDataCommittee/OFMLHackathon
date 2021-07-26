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

#ifndef SMARTREDIS_COMMANDREPLY_H
#define SMARTREDIS_COMMANDREPLY_H

#include "stdlib.h"
#include <sw/redis++/redis++.h>
#include <iostream>

namespace SmartRedis {

class CommandReply;

/*!
*   \typedef Redis++ command reply type
*/
typedef std::unique_ptr<redisReply, sw::redis::ReplyDeleter>
        RedisReplyUPtr;

///@file
/*!
*   \brief The CommandReply class stores and processes Command
*          replies.
*   \details The command reply was built around processing
*          redis-plus-plus replies.  The redis-plus-plus
*          generic command execution returns a unique
*          ptr to a hiredis redisReply.  The destructor
*          of the unique ptr will call the hiredis
*          redisReply destructor, which will recursively
*          delete all replies.  As a result, we only
*          construct the CommandReply through move
*          operations because copy operations cannot
*          be performed for a unique_ptr.  This class
*          also makes it easy to process the reply because
*          the CommandReply class masks the type
*          change from RedisReplyUPtr to redisReply
*          as you inspect sub replies.
*/
class CommandReply {

    public:

        /*!
        *   \brief Default CommandReply constructor
        */
        CommandReply() = default;

        /*!
        *   \brief Default CommandReply destructor
        */
        ~CommandReply() = default;

        /*!
        *   \brief CommandReply copy constructor
        *          not available
        */
        CommandReply(const CommandReply& reply) = delete;

        /*!
        *   \brief CommandReply copy assignment operator
        *          not available
        */
        CommandReply& operator=(const CommandReply& reply) = delete;

        /*!
        *   \brief Move constructor with RedisReplyUPtr
        *          as input.
        *   \param reply The RedisReplyUPtr for construction
        */
        CommandReply(RedisReplyUPtr&& reply);

        /*!
        *   \brief Move constructor with redisReply as input
        *   \param reply The redisReply for construction
        */
        CommandReply(redisReply*&& reply);

        /*!
        *   \brief Move constructor with CommandReply as input
        *   \param reply The CommandReply for construction
        */
        CommandReply(CommandReply&& reply);

        /*!
        *   \brief Move assignment operator with RedisReplyUPtr
        *          as input.
        *   \param reply The RedisReplyUPtr for construction
        *   \returns CommandReply reference
        */
        CommandReply& operator=(RedisReplyUPtr&& reply);

        /*!
        *   \brief Move assignment operator with redisReply
        *          as input.
        *   \param reply The redisReply for construction
        *   \returns CommandReply reference
        */
        CommandReply& operator=(redisReply*&& reply);

        /*!
        *   \brief Move assignment operator with CommandReply
        *          as input.
        *   \param reply The CommandReply for construction
        *   \returns CommandReply reference
        */
        CommandReply& operator=(CommandReply&& reply);

        /*!
        *   \brief Index operator for CommandReply
        *          that will return the indexed element
        *          of the CommandReply if there are multiple
        *          elements
        *   \param index Index to retrieve
        *   \returns The indexed command reply
        */
        CommandReply operator[](int index);

        /*!
        *   \brief Get the string field of the reply
        *   \returns C-str for the CommandReply field
        *   \throw std::runtime_error if the CommandReply
        *          does not have a string field
        */
        char* str();

        /*!
        *   \brief Get the integer field of the reply
        *   \returns long long for the integer CommandReply field
        *   \throw std::runtime_error if the CommandReply
        *          does not have an integer field
        */
        long long integer();

        /*!
        *   \brief Get the double field of the reply
        *   \returns The double CommandReply field
        *   \throw std::runtime_error if the CommandReply
        *          does not have a double field
        */
        double dbl();

        /*!
        *   \brief Get the length of the CommandReply string field
        *   \returns The length of the CommandReply string field
        *   \throw std::runtime_error if the CommandReply
        *          does not have a string field
        */
        size_t str_len();

        /*!
        *   \brief Get the number of elements in the CommandReply
        *   \returns The number of elements in the CommandReply
        *   \throw std::runtime_error if the CommandReply
        *          does not have a multiple elements (i.e.
        *          not a reply array).
        */
        size_t n_elements();

        /*!
        *   \brief Return the number of errors in the CommandReply and
        *          and any nested CommandReply
        *   \returns The number of errors in the CommandReply and
        *            nested CommandReply
        */
        int has_error();

        /*!
        *   \brief This will print any errors in the CommandReply
        *          or nested CommandReply.
        */
        void print_reply_error();

        /*!
        *   \brief Return the type of the CommandReply
        *          in the form of a string.
        *   \returns String form of the reply type
        */
        std::string redis_reply_type();

        /*!
        *   \brief Print the reply structure of the CommandReply
        */
        void print_reply_structure(std::string index_tracker="reply[0]");


    private:

        /*!
        *   \brief CommandReply constructor from a redisReply.
        *   \param reply redisReply for construction
        */
        CommandReply(redisReply* reply);

        /*!
        *   \brief RedisReplyUPtr that can hold redis reply data
        */
        RedisReplyUPtr _uptr_reply;

        /*!
        *   \brief redisReply that can hold redis reply data
        */
        redisReply* _reply;

        /*!
        *   \brief Helper function to print the redis reply
        *          structure.
        *   \param reply redisReply to print structure
        *   \param index_tracker String representing previous
        *                        levels of the nested structure
        */
        void _print_nested_reply_structure(redisReply* reply,
                                           std::string index_tracker);
};

} //namespace SmartRedis

#endif //SMARTREDIS_COMMANDREPLY_H
